"""Rules, accounting, learning, reproducibility and audit regression tests."""
import contextlib
import csv
import io
import itertools
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import gzip

from poker_experiment import (Agent, Config, Deal, EquityEstimator, audit_run,
                             hand_rank, legal_actions, metrics, play_hand,
                             run_experiment, seed_for)


def cards(text):
    return ['23456789TJQKA'.index(c[0])*4+'cdhs'.index(c[1]) for c in text.split()]


def five_reference(cs):
    """Independent five-card classifier to check the direct seven-card evaluator."""
    ranks = sorted([c//4+2 for c in cs],reverse=True)
    counts = {r:ranks.count(r) for r in set(ranks)}
    groups = sorted(counts,key=lambda r:(counts[r],r),reverse=True)
    pattern = sorted(counts.values(),reverse=True)
    flush = len({c%4 for c in cs}) == 1
    distinct = sorted(counts,reverse=True)
    high = (5 if distinct == [14,5,4,3,2] else
            distinct[0] if len(distinct)==5 and distinct[0]-distinct[-1]==4 else 0)
    if flush and high: return (8,high)
    if pattern == [4,1]: return (7,*groups)
    if pattern == [3,2]: return (6,*groups)
    if flush: return (5,*ranks)
    if high: return (4,high)
    if pattern == [3,1,1]: return (3,*groups)
    if pattern == [2,2,1]: return (2,*groups)
    if pattern == [2,1,1,1]: return (1,*groups)
    return (0,*ranks)


class Scripted(Agent):
    def __init__(self, actions):
        super().__init__('scripted')
        self.actions = iter(actions)

    def choose(self,state,legal,equity,odds,rng,epsilon=0):
        a = next(self.actions, 'check' if 'check' in legal else 'call')
        if a not in legal:
            raise AssertionError(f'Script requires {a}, but legal actions are {legal}')
        return a


class ConstantEquity:
    def estimate(self,hole,board):
        return .3


class PokerTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(equity_samples=4)
        self.deal = Deal.from_seed(123,0)

    def hand(self,a,b,stacks=None,deal=None,button=0):
        stacks = [100000,100000] if stacks is None else stacks
        return play_hand([a,b],stacks,button,deal or self.deal,
                         ConstantEquity(),self.cfg,456)

    def test_ranking_categories_and_kickers(self):
        examples = [
            ('As Ks Qs Js Ts 2c 3d', (8,14)),
            ('As Ad Ah Ac Ks 2c 3d', (7,14,13)),
            ('As Ad Ah Ks Kd Kh 2c', (6,14,13)),
            ('As Js 9s 5s 2s Kd Qd', (5,14,11,9,5,2)),
            ('As 2d 3h 4c 5s Kd Qd', (4,5)),
            ('As Ad Ah Ks Qd 2c 3d', (3,14,13,12)),
            ('As Ad Ks Kd Qh Qd 2c', (2,14,13,12)),
            ('As Ad Ks Qd Jh 2c 3d', (1,14,13,12,11)),
            ('As Kd Qh Jc 9s 3d 2c', (0,14,13,12,11,9))]
        for text,expected in examples:
            self.assertEqual(hand_rank(cards(text)),expected)
        with self.assertRaises(ValueError):
            hand_rank([0]*7)

    def test_seven_card_evaluator_against_all_five_subsets(self):
        rng = random.Random(887)
        for _ in range(1200):
            hand = rng.sample(range(52),7)
            self.assertEqual(hand_rank(hand),max(five_reference(cs)
                             for cs in itertools.combinations(hand,5)))

    def test_deal_unique_and_deterministic(self):
        self.assertEqual(self.deal,Deal.from_seed(123,0))
        self.assertEqual(len(set(sum((list(h) for h in self.deal.hole),[])
                                 +list(self.deal.board)+list(self.deal.burns))),12)

    def test_limp_preserves_big_blind_option(self):
        h = self.hand(Scripted(['call','fold']),Scripted(['raise_min']))
        self.assertEqual([a['action'] for a in h['actions']],['call','raise_min','fold'])
        self.assertEqual(h['net_chips'],[-1000,1000])

    def test_checkdown_order_and_both_players_act(self):
        h = self.hand(Scripted(['call']),Scripted([]))
        self.assertEqual([a['actor'] for a in h['actions']],[0,1,1,0,1,0,1,0])
        self.assertEqual([a['street'] for a in h['actions']],
                         ['preflop']*2+['flop']*2+['turn']*2+['river']*2)
        self.assertEqual([len(a['board']) for a in h['actions']],[0,0,3,3,4,4,5,5])

    def test_minimum_raise_and_pot_raise(self):
        legal = legal_actions([99500,99000],[500,1000],0,1000,1000,1500)
        self.assertEqual(legal,{'fold':0,'call':500,'raise_min':1500,
                                'raise_pot':2500,'all_in':99500})
        later = legal_actions([90000,90000],[0,0],0,0,1000,20000)
        self.assertEqual(later['raise_pot'],20000)
        # Duplicates are suppressed, preserving uniformity over distinct choices.
        short = legal_actions([1200,99000],[500,1000],0,1000,1000,1500)
        self.assertEqual(list(short),['fold','call','raise_min'])
        self.assertEqual(short['raise_min'],1200)

    def test_call_after_raise_closes_street(self):
        h = self.hand(Scripted(['raise_min']),Scripted(['call']))
        self.assertEqual(len([a for a in h['actions'] if a['street']=='preflop']),2)

    def test_raise_minimum_tracks_previous_full_raise(self):
        h = self.hand(Scripted(['raise_min','raise_min']),
                      Scripted(['raise_pot','call']))
        actions = h['actions'][:4]
        self.assertEqual([a['action'] for a in actions],
                         ['raise_min','raise_pot','raise_min','call'])
        self.assertEqual(actions[2]['street_bets_before'],[2000,6000])
        self.assertEqual(actions[2]['chips_added'],8000)

    def test_learning_can_choose_low_equity_raise(self):
        learner = Agent()
        state = (0,)
        learner.q[(state,'raise_min')] = 12
        self.assertEqual(learner.choose(state,{'check':0,'raise_min':1000},
                                        .10,0,random.Random(1)), 'raise_min')
        learner.q[(state,'check')] = 20
        self.assertEqual(learner.choose(state,{'check':0,'raise_min':1000},
                                        .10,0,random.Random(1)), 'check')

    def test_unmatched_all_in_refunded(self):
        h = self.hand(Scripted(['all_in']),Scripted(['call']),[100000,10000])
        self.assertEqual(h['contributions'],[100000,10000])
        self.assertEqual(h['uncalled_refunds'],[90000,0])
        self.assertEqual(h['contested_pot'],20000)
        self.assertEqual(len(h['actions']),2)
        self.assertEqual(sum(h['ending_stacks']),110000)
        self.assertEqual(sum(h['net_chips']),0)
        self.assertLessEqual(abs(h['net_chips'][0]),10000)

    def test_short_all_in_fold_keeps_only_opponent_contribution(self):
        h = self.hand(Scripted(['raise_min']),Scripted(['fold']),[1700,100000])
        self.assertEqual(h['uncalled_refunds'],[700,0])
        self.assertEqual(h['net_chips'],[1000,-1000])

    def test_all_in_blinds_and_short_call(self):
        for stacks in ([100,100000],[100000,100],[500,1000],[100,200]):
            h = self.hand(Scripted([]),Scripted([]),list(stacks))
            self.assertEqual(sum(h['net_chips']),0)
            self.assertGreaterEqual(min(h['ending_stacks']),0)
            self.assertLessEqual(abs(h['net_chips'][0]),min(stacks))
        h = self.hand(Scripted(['call']),Scripted(['call']),[750,100000])
        self.assertEqual(h['contributions'],[750,1000])
        self.assertEqual(h['uncalled_refunds'],[0,250])

    def test_showdown_tie_split(self):
        deal = Deal((tuple(cards('2c 3d')),tuple(cards('4c 5d'))),
                    tuple(cards('As Ks Qs Js Ts')),tuple(cards('6c 7c 8c')))
        h = self.hand(Scripted(['call']),Scripted([]),deal=deal)
        self.assertIsNone(h['winner'])
        self.assertEqual(h['net_chips'],[0,0])

    def test_learning_sample_average_first_visit(self):
        a = Agent()
        state = (1,)
        a.learn([(state,'call'),(state,'call')],-10)
        self.assertEqual(a.q[(state,'call')],-10)
        self.assertEqual(a.visits[(state,'call')],1)
        a.learn([(state,'call')],30)
        self.assertEqual(a.q[(state,'call')],10)
        self.assertEqual(a.visits[(state,'call')],2)

    def test_decision_denominator_and_all_chip_changes(self):
        h = self.hand(Scripted(['raise_min','fold']),Scripted(['raise_min']))
        m = metrics([h],1000,0)
        self.assertEqual(m['decisions'],2)
        self.assertEqual(m['low_equity_bets_or_raises'],1)
        self.assertEqual(m['low_equity_aggression_per_decision_percent'],50)
        fold = self.hand(Scripted(['fold']),Scripted([]))
        self.assertEqual(metrics([fold],1000,0)['net_bb_per_10'],-5)

    def test_equity_only_visible_cards_and_repeatable(self):
        estimator = EquityEstimator(42,100)
        self.assertEqual(estimator.estimate(cards('As Ad'),[]),
                         EquityEstimator(42,100).estimate(cards('As Ad'),[]))
        self.assertGreater(estimator.estimate(cards('As Ad'),[]),.65)
        # Royal flush on board: everyone ties regardless of the two private cards.
        self.assertEqual(estimator.estimate(cards('2c 3c'),cards('As Ks Qs Js Ts')),.5)

    def test_randomized_chip_accounting_and_legal_choices(self):
        rng = random.Random(171)
        for i in range(250):
            stacks = [rng.randint(1,100000),rng.randint(1,100000)]
            h = self.hand(Agent('random'),Agent('random'),stacks,
                          Deal.from_seed(i,i%2),i%2)
            self.assertEqual(sum(h['net_chips']),0)
            for a in h['actions']:
                self.assertEqual(a['legal_actions'][a['action']],a['chips_added'])
                self.assertLessEqual(a['chips_added'],a['stacks_before'][a['actor']])

    def test_full_run_replay_and_reproducibility(self):
        config = Config(training_hands=20,rounds=3,equity_samples=4,bootstrap_samples=100)
        with tempfile.TemporaryDirectory() as tmp:
            first,second = Path(tmp)/'one',Path(tmp)/'two'
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                result = run_experiment(config,first)
                run_experiment(config,second)
                self.assertEqual(audit_run(first),result)
            self.assertNotEqual(result['learning']['initial_policy_sha256'],
                                result['learning']['trained_policy_sha256'])
            for p in first.iterdir():
                self.assertEqual(p.read_bytes(),(second/p.name).read_bytes())
            hands = [json.loads(line) for line in (first/'hands.jsonl').read_text().splitlines()]
            before = [h for h in hands if h['phase']=='before']
            after = [h for h in hands if h['phase']=='after']
            self.assertEqual([h['hole_cards'] for h in before],[h['hole_cards'] for h in after])
            self.assertEqual([h['full_board'] for h in before],[h['full_board'] for h in after])
            with (first/'hands.jsonl').open('a') as f:
                f.write('\n')
            with self.assertRaisesRegex(AssertionError,'checksum'):
                audit_run(first)

    def test_streamed_compressed_shards_match_plain_run(self):
        config = Config(training_hands=23,rounds=3,equity_samples=4,bootstrap_samples=100)
        with tempfile.TemporaryDirectory() as tmp:
            plain,compressed = Path(tmp)/'plain',Path(tmp)/'compressed'
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                expected = run_experiment(config,plain)
                with patch('poker_experiment.COMPRESS_AT_HANDS',1),patch('poker_experiment.LOG_SHARD_HANDS',7):
                    actual = run_experiment(config,compressed)
                self.assertEqual(actual,expected)
                self.assertGreater(len(list(compressed.glob('hands-*.jsonl.gz'))),1)
                self.assertEqual(audit_run(compressed),expected)
                self.assertEqual(audit_run(compressed,replay=False),expected)
            joined = ''.join(gzip.open(p,'rt').read() for p in sorted(compressed.glob('hands-*.jsonl.gz')))
            self.assertEqual(joined,(plain/'hands.jsonl').read_text())
            self.assertEqual(gzip.open(compressed/'rounds.csv.gz','rt').read(),(plain/'rounds.csv').read_text())

    def test_equity_cache_eviction_preserves_values(self):
        with patch('poker_experiment.EQUITY_CACHE_LIMIT',1):
            e = EquityEstimator(42,4)
            original = e.estimate(cards('As Ad'),[])
            e.estimate(cards('2c 3c'),[])
            self.assertLessEqual(len(e.cache),1)
            self.assertEqual(e.estimate(cards('As Ad'),[]),original)


if __name__ == '__main__':
    unittest.main()
