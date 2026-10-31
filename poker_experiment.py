"""Local heads-up Hold'em experiment with first-visit Monte Carlo control.

All results are measured from dealt hands; no target statistics are supplied.
Python 3.11+, standard library only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys

DECK = tuple(range(52))
STREETS = ('preflop', 'flop', 'turn', 'river')
ALGORITHM_VERSION = '1.0'


def card_text(card: int) -> str:
    return '23456789TJQKA'[card // 4] + 'cdhs'[card % 4]


def hand_rank(cards) -> tuple[int, ...]:
    """Exact best-five ordering for 5–7 distinct cards; larger is stronger."""
    if not 5 <= len(cards) <= 7 or len(set(cards)) != len(cards):
        raise ValueError('Expected 5–7 distinct cards')
    if any(c not in DECK for c in cards):
        raise ValueError('Invalid card')
    ranks = [c // 4 + 2 for c in cards]
    counts = Counter(ranks)
    def straight(rs):
        rs = set(rs)
        if 14 in rs:
            rs.add(1)
        return next((h for h in range(14, 4, -1)
                     if all(h-i in rs for i in range(5))), 0)
    suits = [[c // 4 + 2 for c in cards if c % 4 == s] for s in range(4)]
    flush = next((s for s in suits if len(s) >= 5), None)
    if flush is not None and (high := straight(flush)):
        return (8, high)
    groups = sorted(((n, r) for r, n in counts.items()), reverse=True)
    if groups[0][0] == 4:
        return (7, groups[0][1], max(r for r in counts if r != groups[0][1]))
    trips = sorted((r for r, n in counts.items() if n >= 3), reverse=True)
    if trips:
        pairs = sorted((r for r, n in counts.items() if n >= 2 and r != trips[0]), reverse=True)
        if pairs:
            return (6, trips[0], pairs[0])
    if flush is not None:
        return (5, *sorted(flush, reverse=True)[:5])
    if high := straight(counts):
        return (4, high)
    if trips:
        return (3, trips[0], *sorted((r for r in counts if r != trips[0]), reverse=True)[:2])
    pairs = sorted((r for r, n in counts.items() if n == 2), reverse=True)
    if len(pairs) >= 2:
        return (2, *pairs[:2], max(r for r in counts if r not in pairs[:2]))
    if pairs:
        return (1, pairs[0], *sorted((r for r in counts if r != pairs[0]), reverse=True)[:3])
    return (0, *sorted(counts, reverse=True)[:5])


def seed_for(seed: int, *parts) -> int:
    payload = json.dumps([seed, *parts], separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], 'big')


class EquityEstimator:
    """Uses only own cards and exposed board, never the actual opponent hand."""
    def __init__(self, seed: int, samples: int):
        self.seed, self.samples = seed, samples
        self.cache = {}

    def estimate(self, hole, board):
        key = (tuple(sorted(hole)), tuple(sorted(board)))
        if key in self.cache:
            return self.cache[key]
        rng = random.Random(seed_for(self.seed, 'equity', *key))
        remaining = [c for c in DECK if c not in hole and c not in board]
        score = 0.0
        for _ in range(self.samples):
            draw = rng.sample(remaining, 7-len(board))
            final = list(board) + draw[2:]
            own, other = hand_rank(list(hole)+final), hand_rank(draw[:2]+final)
            score += 1 if own > other else .5 if own == other else 0
        value = score / self.samples
        self.cache[key] = value
        return value


@dataclass(frozen=True)
class Config:
    seed: int = 42
    training_hands: int = 1000
    rounds: int = 50
    hands_per_round: int = 3
    stack: int = 100000
    big_blind: int = 1000
    equity_samples: int = 64
    baseline: str = 'random'
    epsilon_start: float = .30
    epsilon_end: float = .05
    bootstrap_samples: int = 5000

    def validate(self):
        if min(self.training_hands, self.rounds, self.hands_per_round,
               self.equity_samples, self.bootstrap_samples) < 1:
            raise ValueError('Counts must be positive')
        if self.big_blind < 2 or self.big_blind % 2 or self.stack < self.big_blind:
            raise ValueError('Use an even positive big blind and stack >= big blind')
        if self.baseline not in ('random', 'rules'):
            raise ValueError('Unknown baseline')
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError('Require 0 <= epsilon_end <= epsilon_start <= 1')


def legal_actions(stacks, bets, actor, current, min_raise, pot):
    """Amounts are chips added now. Duplicate raise sizes appear only once."""
    other = 1-actor
    call = max(0, current-bets[actor])
    actions = {'fold': 0, 'call': min(call, stacks[actor])} if call else {'check': 0}
    # If the opponent is already all-in, only match or fold.
    if stacks[other] == 0 or stacks[actor] <= call:
        return actions
    maximum = bets[actor]+stacks[actor]
    minimum = current+min_raise
    targets = [('raise_min', minimum),
               ('raise_pot', current+pot+call),
               ('all_in', maximum)]
    seen = set()
    for name, target in targets:
        target = min(maximum, max(minimum, target))
        if target > current and target not in seen:
            actions[name] = target-bets[actor]
            seen.add(target)
    return actions


def state_key(street, equity, odds, button, stacks, actor, pot, legal):
    # Include the legal action mask so unavailable actions are never selected.
    return (street, min(4, int(equity*5)), int(actor == button),
            0 if odds == 0 else 1 if odds <= .2 else 2 if odds <= .4 else 3,
            min(2, int(min(stacks)/max(1, pot))), tuple(legal))


class Agent:
    def __init__(self, kind='learner'):
        self.kind = kind
        self.q = {}
        self.visits = {}
        self.update_count = 0

    def choose(self, state, legal, equity, odds, rng, epsilon=0):
        if self.kind == 'random':
            return rng.choice(list(legal))
        if self.kind == 'rules':
            if 'fold' in legal and equity < odds+.05:
                return 'fold'
            if equity >= .70 and 'raise_pot' in legal:
                return 'raise_pot'
            if equity >= .60 and 'raise_min' in legal:
                return 'raise_min'
            return 'check' if 'check' in legal else 'call'
        if rng.random() < epsilon:
            return rng.choice(list(legal))
        values = {a: self.q.get((state,a), 0.0) for a in legal}
        best = max(values.values())
        return rng.choice([a for a in legal if values[a] == best])

    def learn(self, trace, reward_bb):
        # First visit to each state-action pair in this episode. Full, unclipped
        # net hand reward, gamma=1; incremental sample-average update.
        seen = set()
        for state, action in trace:
            key = (state,action)
            if key in seen:
                continue
            seen.add(key)
            n = self.visits.get(key,0)+1
            self.visits[key] = n
            old = self.q.get(key,0.0)
            self.q[key] = old+(reward_bb-old)/n
            self.update_count += 1

    def snapshot(self):
        entries = [{'state': s, 'action': a, 'value_bb': v,
                    'visits': self.visits[(s,a)]}
                   for (s,a),v in sorted(self.q.items(), key=lambda item: repr(item[0]))]
        return {'algorithm':'first-visit Monte Carlo control', 'gamma':1,
                'update_count': self.update_count, 'entries': entries}

    def digest(self):
        return hashlib.sha256(json.dumps(self.snapshot(), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class Deal:
    hole: tuple
    board: tuple
    burns: tuple

    @classmethod
    def from_seed(cls, seed, button):
        deck = list(DECK)
        random.Random(seed).shuffle(deck)
        hole = [[],[]]
        for _ in range(2):
            for seat in (1-button,button):
                hole[seat].append(deck.pop())
        burns, board = [deck.pop()], [deck.pop() for _ in range(3)]
        burns.append(deck.pop())
        board.append(deck.pop())
        burns.append(deck.pop())
        board.append(deck.pop())
        return cls(tuple(tuple(h) for h in hole),tuple(board),tuple(burns))


def play_hand(agents, stacks, button, deal, estimator, config, action_seed,
              epsilon=0.0, training=False, metadata=None):
    """Play a complete hand and return an auditable private-card hand history."""
    start = stacks.copy()
    if min(start) <= 0:
        raise ValueError('Both players need chips before a hand')
    total = sum(start)
    contributed = [0,0]
    logs, trace = [], []
    bets = [0,0]
    def put(actor, amount):
        if not 0 <= amount <= stacks[actor]:
            raise AssertionError('Illegal contribution')
        stacks[actor] -= amount
        contributed[actor] += amount
        bets[actor] += amount
        if sum(stacks)+sum(contributed) != total:
            raise AssertionError('Chip conservation failed')
    put(button, min(config.big_blind//2, stacks[button]))
    put(1-button, min(config.big_blind, stacks[1-button]))
    folded = None
    action_counts = [0,0]
    for street in range(4):
        if street:
            bets = [0,0]
        exposed = [] if street == 0 else list(deal.board[:street+2])
        # An all-in blind must be matched/folded if the other seat still owes;
        # otherwise no player can make another contested bet.
        if 0 in stacks:
            live = 0 if stacks[0] else 1
            if stacks[live] == 0 or bets[live] >= bets[1-live]:
                break
            pending = {live}
        else:
            pending = {0,1}
        actor = button if street == 0 else 1-button
        current = max(config.big_blind, max(bets)) if street == 0 else 0
        last_raise = config.big_blind
        while pending:
            if actor not in pending:
                actor = 1-actor
                continue
            if stacks[actor] == 0:
                pending.discard(actor)
                actor = 1-actor
                continue
            other = 1-actor
            # A short all-in blind is matched only up to its actual size.
            if stacks[other] == 0:
                current = max(bets)
            to_call = max(0,current-bets[actor])
            pot = sum(contributed)
            legal = legal_actions(stacks,bets,actor,current,last_raise,pot)
            strength = estimator.estimate(deal.hole[actor],exposed)
            call_cost = min(to_call,stacks[actor])
            unmatchable = max(0,contributed[other]-contributed[actor]-call_cost)
            odds = call_cost/(pot+call_cost-unmatchable) if call_cost else 0
            key = state_key(street,strength,odds,button,stacks,actor,pot,legal)
            rng = random.Random(seed_for(action_seed,street,actor,action_counts[actor]))
            action = agents[actor].choose(key,legal,strength,odds,rng,epsilon if actor == 0 else 0)
            if action not in legal:
                raise AssertionError('Agent chose illegal action')
            amount = legal[action]
            action_counts[actor] += 1
            aggressive = action.startswith('raise') or action == 'all_in'
            logs.append({'street':STREETS[street],'actor':actor,
                         'board':[card_text(c) for c in exposed],
                         'pot_before':pot,'stacks_before':stacks.copy(),
                         'street_bets_before':bets.copy(),'to_call':to_call,
                         'legal_actions':legal.copy(),'action':action,'chips_added':amount,
                         'estimated_equity':strength,'equity_samples':config.equity_samples,
                         'low_equity_aggression':aggressive and strength < .40,
                         'state':key})
            if actor == 0 and training:
                trace.append((key,action))
            if action == 'fold':
                folded = actor
                break
            pending.discard(actor)
            if aggressive:
                new_total = bets[actor]+amount
                increment = new_total-current
                put(actor,amount)
                if increment >= last_raise:
                    last_raise = increment
                current = new_total
                pending = {other} if stacks[other] else set()
            elif action == 'call':
                put(actor,amount)
            # A limp removes the button only; the BB retains its option.
            if 0 in stacks:
                live = 0 if stacks[0] else 1
                if stacks[live] == 0 or bets[live] >= bets[1-live]:
                    pending.clear()
                else:
                    pending = {live}
            actor = other
        if folded is not None or 0 in stacks:
            break
    pot = sum(contributed)
    refunds = [0,0]
    unmatched = abs(contributed[0]-contributed[1])
    if unmatched:
        high = 0 if contributed[0] > contributed[1] else 1
        refunds[high] = unmatched
        stacks[high] += unmatched
        pot -= unmatched
    payouts = [0,0]
    showdown_ranks = None
    if folded is not None:
        winner = 1-folded
        payouts[winner] = pot
        reason = 'fold'
    else:
        showdown_ranks = [hand_rank(list(h)+list(deal.board)) for h in deal.hole]
        if showdown_ranks[0] == showdown_ranks[1]:
            winner = None
            payouts = [pot//2,pot//2]
            payouts[1-button] += pot%2
        else:
            winner = int(showdown_ranks[1] > showdown_ranks[0])
            payouts[winner] = pot
        reason = 'showdown'
    for seat in range(2):
        stacks[seat] += payouts[seat]
    delta = [stacks[s]-start[s] for s in range(2)]
    if sum(stacks) != total or sum(delta) or min(stacks) < 0:
        raise AssertionError('Invalid final chip accounting')
    if training:
        agents[0].learn(trace,delta[0]/config.big_blind)
    return {**(metadata or {}),'button':button,'starting_stacks':start,
            'hole_cards':[[card_text(c) for c in h] for h in deal.hole],
            'full_board':[card_text(c) for c in deal.board],
            'burn_cards':[card_text(c) for c in deal.burns],
            'actions':logs,'contributions':contributed,'uncalled_refunds':refunds,
            'contested_pot':pot,'payouts':payouts,'ending_stacks':stacks.copy(),
            'net_chips':delta,'winner':winner,'ending_reason':reason,
            'showdown_ranks':showdown_ranks}


def metrics(hands, bb, seat):
    acts = [a for h in hands for a in h['actions'] if a['actor'] == seat]
    bluffs = sum(a['low_equity_aggression'] for a in acts)
    aggressive = sum(a['action'].startswith('raise') or a['action']=='all_in' for a in acts)
    net = sum(h['net_chips'][seat] for h in hands)
    wins = sum(h['winner'] == seat for h in hands)
    ties = sum(h['winner'] is None for h in hands)
    return {'hands':len(hands),'decisions':len(acts),'low_equity_bets_or_raises':bluffs,
            'aggressive_actions':aggressive,
            'low_equity_aggression_per_decision_percent':100*bluffs/len(acts) if acts else None,
            'low_equity_share_of_aggressive_actions_percent':100*bluffs/aggressive if aggressive else None,
            'hands_with_low_equity_aggression':sum(any(a['low_equity_aggression'] and a['actor']==seat
                                                     for a in h['actions']) for h in hands),
            'net_chips':net,'net_bb_per_10':10*net/(bb*len(hands)),
            'hands_won':wins,'hands_tied':ties,'hand_win_percent':100*wins/len(hands)}


def round_rows(hands, bb):
    groups = {}
    for h in hands:
        groups.setdefault((h['phase'],h['round']),[]).append(h)
    rows = []
    for (phase,number), group in groups.items():
        row = {'phase':phase,'round':number,'hands':len(group),
               'bankruptcy_rebuys':sum(h['reset_reason']=='bankruptcy_rebuy' for h in group)}
        for seat,name in ((0,'agent'),(1,'baseline')):
            row.update({name+'_'+key:value for key,value in metrics(group,bb,seat).items()})
        rows.append(row)
    return rows


def percentile(values, p):
    ordered = sorted(values)
    position = (len(ordered)-1)*p
    lower = int(position)
    upper = min(lower+1,len(ordered)-1)
    return ordered[lower]+(ordered[upper]-ordered[lower])*(position-lower)


def paired_change(rounds, config):
    before = [r for r in rounds if r['phase']=='before']
    after = [r for r in rounds if r['phase']=='after']
    diffs = [a['agent_net_bb_per_10']-b['agent_net_bb_per_10'] for b,a in zip(before,after)]
    rng = random.Random(seed_for(config.seed,'bootstrap'))
    means = [statistics.fmean(rng.choices(diffs,k=len(diffs)))
             for _ in range(config.bootstrap_samples)]
    return {'paired_unit':'evaluation round; same deal seeds before and after',
            'rounds':len(diffs),'mean_change_net_bb_per_10':statistics.fmean(diffs),
            'bootstrap_percentile_95_interval':[percentile(means,.025),percentile(means,.975)],
            'bootstrap_resamples':config.bootstrap_samples,
            'p_value':None,
            'scope':'Conditional on this one training run and opponent. Does not estimate variability across training seeds.'}


def run_experiment(config, output):
    config.validate()
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Output directory must be empty; choose a new path to preserve earlier runs')
    learner, baseline = Agent(), Agent(config.baseline)
    estimator = EquityEstimator(config.seed,config.equity_samples)
    histories = []
    initial_digest = learner.digest()
    initial_policy = learner.snapshot()
    def phase(name, count, training):
        stacks = [config.stack,config.stack]
        for i in range(count):
            round_id, slot = divmod(i,config.hands_per_round)
            reset_reason = 'none'
            if slot == 0:
                stacks = [config.stack,config.stack]
                reset_reason = 'round_start'
            elif min(stacks) == 0:
                stacks = [config.stack,config.stack]
                reset_reason = 'bankruptcy_rebuy'
            stream = 'train' if training else 'eval'
            deal_seed = seed_for(config.seed,stream,'deal',i)
            action_seed = seed_for(config.seed,stream,'policy',i)
            button = i%2
            deal = Deal.from_seed(deal_seed,button)
            fraction = i/max(1,count-1)
            eps = config.epsilon_start+(config.epsilon_end-config.epsilon_start)*fraction if training else 0
            history = play_hand([learner,baseline],stacks,button,deal,estimator,config,
                                action_seed,eps,training,
                                {'phase':name,'round':round_id+1,'hand':slot+1,
                                 'deal_seed':str(deal_seed),'action_seed':str(action_seed),
                                 'reset_reason':reset_reason,'training_epsilon':eps})
            histories.append(history)
            if (i+1)%100 == 0:
                print(f'{name}: {i+1}/{count} hands',file=sys.stderr,flush=True)
    n_eval = config.rounds*config.hands_per_round
    phase('before',n_eval,False)
    if learner.digest() != initial_digest:
        raise AssertionError('Pre-training evaluation modified policy')
    phase('training',config.training_hands,True)
    trained_policy = learner.snapshot()
    trained_digest = learner.digest()
    phase('after',n_eval,False)
    if learner.digest() != trained_digest:
        raise AssertionError('Evaluation modified trained policy')
    rounds = round_rows(histories,config.big_blind)
    summary = {'algorithm_version':ALGORITHM_VERSION,'config':asdict(config),
               'learning':{'method':'local first-visit Monte Carlo control',
                           'initial_policy_sha256':initial_digest,
                           'trained_policy_sha256':trained_digest,
                           'updated_state_action_pairs':len(learner.q),
                           'updates':learner.update_count,
                           'evaluation_policy_frozen':True},
               'phases':{},'before_after_comparison':paired_change(rounds,config)}
    for phase_name in ('before','training','after'):
        group = [h for h in histories if h['phase']==phase_name]
        summary['phases'][phase_name] = {
            'agent':metrics(group,config.big_blind,0),
            'baseline':metrics(group,config.big_blind,1),
            'bankruptcy_rebuys':sum(h['reset_reason']=='bankruptcy_rebuy' for h in group)}
    # Auditable logs are the source; summaries are calculated from those logs.
    with (output/'hands.jsonl').open('w') as handle:
        for h in histories:
            handle.write(json.dumps(h,separators=(',',':'))+'\n')
    with (output/'rounds.csv').open('w',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=rounds[0].keys())
        writer.writeheader()
        writer.writerows(rounds)
    for filename,data in (('policy_before.json',initial_policy),('policy_after.json',trained_policy),
                          ('summary.json',summary)):
        (output/filename).write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest = {'algorithm_version':ALGORITHM_VERSION,'python':sys.version,
                'source_sha256':source_sha,
                'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(output.iterdir()) if p.is_file()}}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    return summary


def audit_run(output):
    """Recalculate measures and replay every hand against saved policy checkpoints."""
    output = Path(output)
    summary = json.loads((output/'summary.json').read_text())
    config = Config(**summary['config'])
    manifest = json.loads((output/'manifest.json').read_text())
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != manifest['source_sha256']:
        raise AssertionError('Simulator source differs from recorded run')
    for name,digest in manifest['files'].items():
        if hashlib.sha256((output/name).read_bytes()).hexdigest() != digest:
            raise AssertionError(f'File checksum mismatch: {name}')
    original = [json.loads(line) for line in (output/'hands.jsonl').read_text().splitlines()]
    learner, baseline = Agent(),Agent(config.baseline)
    estimator = EquityEstimator(config.seed,config.equity_samples)
    stacks = None
    phase_counts = Counter()
    for h in original:
        name = h['phase']
        i = phase_counts[name]
        if i == 0:
            if name == 'training' and learner.digest() != summary['learning']['initial_policy_sha256']:
                raise AssertionError('Initial policy changed')
            if name == 'after' and learner.digest() != summary['learning']['trained_policy_sha256']:
                raise AssertionError('Trained policy checkpoint mismatch')
        if i%config.hands_per_round == 0:
            stacks = [config.stack,config.stack]
        elif min(stacks) == 0:
            stacks = [config.stack,config.stack]
        if stacks != h['starting_stacks']:
            raise AssertionError('Starting stacks do not follow reset rule')
        stream = 'train' if name == 'training' else 'eval'
        ds = seed_for(config.seed,stream,'deal',i)
        ps = seed_for(config.seed,stream,'policy',i)
        total_hands = config.training_hands if name == 'training' else config.rounds*config.hands_per_round
        epsilon = (config.epsilon_start+(config.epsilon_end-config.epsilon_start)*i/max(1,total_hands-1)
                   if name == 'training' else 0)
        reset = ('round_start' if i%config.hands_per_round == 0 else
                 'bankruptcy_rebuy' if min(previous_ending) == 0 else 'none')
        metadata = {'phase':name,'round':i//config.hands_per_round+1,
                    'hand':i%config.hands_per_round+1,'deal_seed':str(ds),
                    'action_seed':str(ps),'reset_reason':reset,'training_epsilon':epsilon}
        if any(h[k] != v for k,v in metadata.items()):
            raise AssertionError('Hand metadata mismatch')
        if str(ds) != h['deal_seed'] or str(ps) != h['action_seed']:
            raise AssertionError('Seed mismatch')
        result = play_hand([learner,baseline],stacks,i%2,Deal.from_seed(ds,i%2),estimator,
                           config,ps,h['training_epsilon'],name=='training',metadata)
        if json.loads(json.dumps(result)) != h:
            raise AssertionError(f'Hand replay mismatch: {name} #{i+1}')
        previous_ending = stacks.copy()
        phase_counts[name] += 1
    expected = {'before':config.rounds*config.hands_per_round,
                'training':config.training_hands,'after':config.rounds*config.hands_per_round}
    if dict(phase_counts) != expected:
        raise AssertionError('Hand counts mismatch')
    for name in expected:
        for seat,label in ((0,'agent'),(1,'baseline')):
            if metrics([h for h in original if h['phase']==name],config.big_blind,seat) != summary['phases'][name][label]:
                raise AssertionError('Aggregate metric mismatch')
    derived_rounds = round_rows(original,config.big_blind)
    with (output/'rounds.csv').open(newline='') as handle:
        saved = list(csv.DictReader(handle))
    expected_rounds = [{k:'' if v is None else str(v) for k,v in r.items()} for r in derived_rounds]
    if saved != expected_rounds:
        raise AssertionError('Round CSV mismatch')
    if paired_change(derived_rounds,config) != summary['before_after_comparison']:
        raise AssertionError('Paired comparison mismatch')
    if learner.digest() != summary['learning']['trained_policy_sha256']:
        raise AssertionError('Evaluation modified policy')
    if learner.snapshot() != json.loads((output/'policy_after.json').read_text()):
        # JSON turns state tuples into lists; compare canonical serialization.
        if json.loads(json.dumps(learner.snapshot())) != json.loads((output/'policy_after.json').read_text()):
            raise AssertionError('Saved final policy differs from learned policy')
    if json.loads((output/'policy_before.json').read_text()) != Agent().snapshot():
        raise AssertionError('Saved initial policy differs from untrained policy')
    print(f'Audit passed: replayed {len(original)} hands; verified actions, chips, policies and summaries.')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--training-hands',type=int,default=1000)
    parser.add_argument('--rounds',type=int,default=50)
    parser.add_argument('--hands-per-round',type=int,default=3)
    parser.add_argument('--equity-samples',type=int,default=64)
    parser.add_argument('--baseline',choices=('random','rules'),default='random')
    parser.add_argument('--output',type=Path,default=Path('results'))
    parser.add_argument('--audit',type=Path,help='Replay and verify a completed run directory')
    args = parser.parse_args()
    if args.audit:
        audit_run(args.audit)
        return
    config = Config(seed=args.seed,training_hands=args.training_hands,rounds=args.rounds,
                    hands_per_round=args.hands_per_round,equity_samples=args.equity_samples,
                    baseline=args.baseline)
    result = run_experiment(config,args.output)
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__ == '__main__':
    main()
