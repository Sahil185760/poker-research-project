# Heads-up poker reinforcement-learning experiment

This program deals and plays two-player Texas Hold’em, updates an agent from hand outcomes, and measures its performance before and after training. The included run contains complete hand histories, policy checkpoints, and calculated results.

The agent uses local tabular reinforcement learning: its action-value table changes after each training hand. Game play, equity estimation, learning, and evaluation run locally in Python. The method, rules, and recorded configuration are described below.

## Paper

The revised paper (version 2), reporting the 1,000,000-training-hand experiment, is available on [Zenodo](https://doi.org/10.5281/zenodo.23863357). This repository contains the supporting simulator and recorded run data.

## Run

Python 3.9 or later; no external packages or API keys are required. From this directory:

```sh
python3 poker_experiment.py --training-hands 1000000 --seed 42 --output my_run
python3 poker_experiment.py --audit my_run --records-only
python3 -m unittest -v test_poker.py
```

The quick records audit above checks the saved measurements and learning updates. For an optional full replay, including every card deal and equity calculation, run `python3 poker_experiment.py --audit my_run`; this repeats the computation.

A new run needs an empty output directory. Earlier runs are never silently overwritten. To reproduce the included run, use the default configuration and the Python version recorded in `results_1000000_seed42/manifest.json`. Reusing a seed reproduces a run; it does not provide another independent experiment.

The default baseline is **random**: uniform over distinct legal action choices, with no learning. To run the separate rule-based condition:

```sh
python3 poker_experiment.py --baseline rules --output rule_baseline_run
```

The fixed rule baseline folds below pot odds + 0.05, raises pot-sized at equity ≥ 0.70, raises the minimum at equity ≥ 0.60, and otherwise checks or calls. It uses the same equity estimator and never learns. Results from different baseline conditions must be labeled separately.

## Game and experimental design

- A shuffled 52-card deck; two private cards per seat; flop, turn and river with burn cards; best five-card showdown evaluation; no rake.
- Small blind 500 chips, big blind 1,000 chips. The button posts the small blind, acts first preflop and last on later streets. The button alternates each hand.
- Each three-hand round starts with 100,000 chips per player. Balances carry between hands. If either player goes broke before the round ends, both stacks reset to 100,000 before the next hand. This explicit rebuy rule completes all three hands. Rebuys and round resets are logged and excluded from profit.
- The action abstraction permits fold or check, call, minimum raise, pot-sized raise, and all-in when legal. Duplicate amounts are suppressed. Legal no-limit raise sizes between these choices are omitted; this is a simplified action space. Short all-ins are allowed, uncalled excess is refunded, and ties split the contested pot.
- The untrained agent first plays 50 evaluation rounds of three hands (150 hands). It then learns on exactly 1,000,000 separate training hands. Its final policy is frozen and evaluated on 50 rounds of three hands again.
- Before and after evaluation use the same deal seeds, cards and button positions, and the same opponent type. Training uses a separate random stream. Cards, policy choices and equity sampling use separate deterministic streams. Later balances and action paths can differ between evaluations; pairing is therefore by whole round.

## What the agent learns

The learner uses first-visit Monte Carlo control with an epsilon-greedy training policy. Its action values start at zero; ties are broken randomly. Training exploration decreases linearly from 0.30 to 0.05 over 1,000,000 hands. Evaluation uses zero exploration, with random tie-breaking. Every legal action remains available, including a raise with low estimated equity.

A state combines the street, one of five equity bins, button position, pot-odds bin, effective-stack-to-pot bin, and legal action choices. After a hand, the reward is the full net chip change divided by 1,000. Each state-action pair is updated once per hand by the incremental mean of observed rewards (`Q ← Q + (reward − Q)/visits`, discount factor 1). No reward is clipped. State aggregation omits parts of betting history and opponent tendencies, so it cannot represent a complete poker strategy or guarantee optimal play.

At each decision, 64 Monte Carlo draws estimate showdown equity against a uniformly random opponent hand and unknown future community cards. The estimate uses only the acting player’s private cards and the board already exposed. It never reads the actual opposing private cards or hidden future board. The estimate is deterministically seeded by the visible cards. Its cache is cleared at 100,000 entries to bound memory; clearing the cache does not change computed estimates. This coarse estimate is noisy and does not model the opponent’s betting-conditioned range.

The learning algorithm follows [Sutton and Barto’s description of first-visit Monte Carlo control](https://www.incompleteideas.net/book/bookdraft2018mar21.pdf), adapted to this small state abstraction. The betting implementation follows the [heads-up blind and betting rules](https://www.pokerstars.com/help/articles/poker-rules-master/229151/) subject to the stated action abstraction and rebuy rule.

## Measures and uncertainty

“Low-equity aggression” is a bet or raise with estimated showdown equity below 0.40. The main frequency is the number of those actions divided by **all decisions** made by that agent. Counts and denominators are saved. This proxy includes semi-bluffs and does not infer the agent’s intent. A second measure divides by aggressive actions only. These denominators must not be interchanged or described as percentages of hands. The number of hands containing such an action is also saved.

Net BB/10 = `10 × total net chips / (1,000 × hands played)`. Every gain and loss counts, including the small blind. A hand win percentage is reported separately: wins divided by hands, with ties counted separately. Head-to-head net returns are exactly opposite and are not independent samples.

The before-to-after comparison subtracts each before-round return from its matched after-round return. An exploratory 95% percentile interval resamples those 50 paired round differences 5,000 times. This describes evaluation variation conditional on this one trained policy and opponent; it does not estimate variation across independent training runs. No p-value is asserted. One training run cannot establish a reliable benefit across training seeds; more frequent bluffing alone would not establish improvement.

## Included run: seed 42

These are the outcomes of the recorded **1,000,000-training-hand** run, not a selected best seed. Each evaluation remains 150 hands.

| Agent measure | Before training | After 1,000,000 training hands |
| --- | ---: | ---: |
| Evaluation hands | 150 | 150 |
| Net chips | -751,500 | 1,148,500 |
| Net BB/10 | -50.10 | 76.57 |
| Hands won | 68 | 82 |
| Hand win percentage | 45.33% | 54.67% |
| Tied hands | 3 | 4 |
| Low-equity bets or raises / decisions | 34 / 241 | 46 / 366 |
| Low-equity aggression per decision | 14.11% | 12.57% |

The paired mean change is **+126.67 BB/10**, with an exploratory 95% bootstrap interval of **+30.35 to +226.07 BB/10**. This interval is conditional on one trained policy. It does not estimate variation across independent training seeds. A larger training budget does not enlarge the evaluation set or supply independent replications. Seats alternate deterministically; they are not randomized.

The learner made 2,216,686 state-action updates involving 2,732 distinct pairs. Its policy hashes changed during training and stayed unchanged during evaluation. The records audit checks all 1,000,300 logged hands, recomputes counts and chip returns, reconstructs learning from recorded states/actions, and checks the policy checkpoints, round summaries, and paired interval. It does not independently recompute every deal or equity estimate; the optional full replay does that.

## Files

The three source files are `poker_experiment.py`, `test_poker.py`, and this README. The included run directory is `results_1000000_seed42`. The program reads the compressed files directly; no manual extraction is required. Large runs use compressed logs and streaming counters to avoid retaining every hand in memory. It contains:

- `hands-00000.jsonl.gz` through `hands-00010.jsonl.gz`: source hand histories containing private cards, board and burns, all decisions, legal actions, equity estimates, bets, refunds, payouts, resets and net changes. These are streamed in ordered gzip parts of at most 100,000 hands. Private cards are recorded for audit after play; agents cannot read the opponent’s cards.
- `rounds.csv.gz`: round-level counts and measurements derived from the histories.
- `policy_before.json` and `policy_after.json`: action-value checkpoints and visit counts.
- `summary.json`: configuration, aggregate measurements, policy hashes and paired comparison.
- `manifest.json`: Python version, source hash and output checksums. Checksums detect file changes; they are not independent proof of when an experiment occurred.

Twenty tests cover hand rankings (including comparison with an independent five-card evaluator over all 21 subsets of 1,200 seven-card hands), blind order, the big blind’s option after a limp, full raise sizes, all-in calls/refunds, ties, chip accounting across varied stacks, learning updates, frequency denominators, deterministic replay, altered-file detection, bounded-cache invariance, and equality of compressed/streamed versus plain outputs. Both audit modes are tested.
