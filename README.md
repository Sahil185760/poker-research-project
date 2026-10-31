# Heads-up poker reinforcement-learning experiment

This program deals and plays two-player Texas Hold’em, updates an agent from hand outcomes, and measures its performance before and after training. The saved results come from running the simulator. There are no supplied target frequencies, target profits, or adjustments to make results match a paper.

This is a new implementation of the paper’s experimental design. It uses **local tabular reinforcement learning**: an action-value table changes after each training hand. It does not fine-tune GPT-4o, modify neural-network weights, change prompts, or call a hosted model. The paper’s model, methods, and numerical findings must be revised to describe this implementation and its measured results before citing this run.

## Run

Python 3.9 or later; no external packages or API keys are required. From this directory:

```sh
python3 poker_experiment.py --seed 42 --output my_run
python3 poker_experiment.py --audit my_run
python3 -m unittest -v test_poker.py
```

A new run needs an empty output directory. Earlier runs are never silently overwritten. To reproduce the included run, use the default configuration and the Python version recorded in `results_seed42/manifest.json`. Reusing a seed reproduces a run; it does not provide another independent experiment.

The paper’s title says random baseline while its methods say rule-based. The default here is **random**: uniform over distinct legal action choices, with no learning. To run the separate rule-based condition:

```sh
python3 poker_experiment.py --baseline rules --output rule_baseline_run
```

The fixed rule baseline folds below pot odds + 0.05, raises pot-sized at equity ≥ 0.70, raises the minimum at equity ≥ 0.60, and otherwise checks or calls. It uses the same equity estimator and never learns. Results from different baseline conditions must be labeled separately.

## Game and experimental design

- A shuffled 52-card deck; two private cards per seat; flop, turn and river with burn cards; best five-card showdown evaluation; no rake.
- Small blind 500 chips, big blind 1,000 chips. The button posts the small blind, acts first preflop and last on later streets. The button alternates each hand.
- Each three-hand round starts with 100,000 chips per player. Balances carry between hands. If either player goes broke before the round ends, both stacks reset to 100,000 before the next hand. This explicit rebuy rule completes all three hands. Rebuys and round resets are logged and excluded from profit.
- The action abstraction permits fold or check, call, minimum raise, pot-sized raise, and all-in when legal. Duplicate amounts are suppressed. Legal no-limit raise sizes between these choices are omitted; this is a simplified action space. Short all-ins are allowed, uncalled excess is refunded, and ties split the contested pot.
- The untrained agent first plays 50 evaluation rounds of three hands (150 hands). It then learns on exactly 1,000 separate training hands. Its final policy is frozen and evaluated on 50 rounds of three hands again.
- Before and after evaluation use the same deal seeds, cards and button positions, and the same opponent type. Training uses a separate random stream. Cards, policy choices and equity sampling use separate deterministic streams. Later balances and action paths can differ between evaluations; pairing is therefore by whole round.

## What the agent learns

The learner uses first-visit Monte Carlo control with an epsilon-greedy training policy. Its action values start at zero; ties are broken randomly. Training exploration decreases linearly from 0.30 to 0.05 over 1,000 hands. Evaluation uses zero exploration, with random tie-breaking. Every legal action remains available, including a raise with low estimated equity.

A state combines the street, one of five equity bins, button position, pot-odds bin, effective-stack-to-pot bin, and legal action choices. After a hand, the reward is the full net chip change divided by 1,000. Each state-action pair is updated once per hand by the incremental mean of observed rewards (`Q ← Q + (reward − Q)/visits`, discount factor 1). No reward is clipped. State aggregation omits parts of betting history and opponent tendencies, so it cannot represent a complete poker strategy or guarantee optimal play.

At each decision, 64 Monte Carlo draws estimate showdown equity against a uniformly random opponent hand and unknown future community cards. The estimate uses only the acting player’s private cards and the board already exposed. It never reads the actual opposing private cards or hidden future board. This coarse estimate is noisy and does not model the opponent’s betting-conditioned range.

The learning algorithm follows [Sutton and Barto’s description of first-visit Monte Carlo control](https://www.incompleteideas.net/book/bookdraft2018mar21.pdf), adapted to this small state abstraction. The betting implementation follows the [heads-up blind and betting rules](https://www.pokerstars.com/help/articles/poker-rules-master/229151/) subject to the stated action abstraction and rebuy rule.

## Measures and uncertainty

“Low-equity aggression” is a bet or raise with estimated showdown equity below 0.40. The main frequency is the number of those actions divided by **all decisions** made by that agent. Counts and denominators are saved. This proxy includes semi-bluffs and does not infer the agent’s intent. A second measure divides by aggressive actions only. These denominators must not be interchanged or described as percentages of hands. The number of hands containing such an action is also saved.

Net BB/10 = `10 × total net chips / (1,000 × hands played)`. Every gain and loss counts, including the small blind. A hand win percentage is reported separately: wins divided by hands, with ties counted separately. Head-to-head net returns are exactly opposite and are not independent samples.

The before-to-after comparison subtracts each before-round return from its matched after-round return. An exploratory 95% percentile interval resamples those 50 paired round differences 5,000 times. This describes evaluation variation conditional on this one trained policy and opponent; it does not estimate variation across independent training runs. No p-value is asserted. One short run cannot establish a reliable training benefit; more frequent bluffing alone would not establish improvement.

## Included run: seed 42

These are the outcomes of the recorded default run, not a selected best seed:

| Agent measure | Before training | After 1,000 training hands |
| --- | ---: | ---: |
| Evaluation hands | 150 | 150 |
| Net chips | −751,500 | −366,000 |
| Net BB/10 | −50.10 | −24.40 |
| Hands won | 68 | 50 |
| Hand win percentage | 45.33% | 33.33% |
| Tied hands | 3 | 3 |
| Low-equity bets or raises / decisions | 34 / 241 | 27 / 207 |
| Low-equity aggression per decision | 14.11% | 13.04% |

The paired mean change is **+25.70 BB/10**, with an exploratory 95% bootstrap interval of **−62.53 to +114.90 BB/10**. The agent still lost overall after training, its hand win percentage fell, and the interval includes zero. This run does not establish improved poker skill or a reliable positive training effect.

The learner made 1,289 state-action updates involving 229 distinct pairs. Its policy hashes changed during training and stayed unchanged during evaluation. The audit replays every one of the 1,300 hands, including the training updates, and checks the stored decisions, stacks, showdown outcomes and summaries.

## Files

The three source files are `poker_experiment.py`, `test_poker.py`, and this README. The included run directory contains:

- `hands.jsonl`: source hand histories, private cards, board and burns, all decisions, legal actions, equity estimates, bets, refunds, payouts, resets and net changes. Private cards are recorded for audit after play; agents cannot read the opponent’s cards.
- `rounds.csv`: round-level counts and measurements derived from the histories.
- `policy_before.json` and `policy_after.json`: action-value checkpoints and visit counts.
- `summary.json`: configuration, aggregate measurements, policy hashes and paired comparison.
- `manifest.json`: Python version, source hash and output checksums. Checksums detect file changes; they are not independent proof of when an experiment occurred.

Eighteen tests cover hand rankings (including comparison with an independent five-card evaluator over all 21 subsets of 1,200 seven-card hands), blind order, the big blind’s option after a limp, full raise sizes, all-in calls/refunds, ties, chip accounting across varied stacks, learning updates, frequency denominators, deterministic replay and altered-file detection.
