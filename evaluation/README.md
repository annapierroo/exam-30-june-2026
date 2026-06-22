# Evaluation

Utilities for evaluating frozen Briscola policies.

## Files

- `match.py`: runs one complete game from four explicit player policies.
- `metrics.py`: computes aggregate metrics from game point differences.
- `suite.py`: runs a learner across evaluation scenarios.

## Match

`play_match` receives:

- player policies;
- environment seed;
- policy seed;
- first player;
- action-selection mode.

It returns final scores, winner, and point margin.

## Metrics

`compute_metrics` reports:

- games;
- win rate;
- draw rate;
- loss rate;
- mean point difference;
- standard error;
- 95% confidence interval.

## Suite

A scenario specifies:

- partner policy;
- next opponent policy;
- previous opponent policy.
