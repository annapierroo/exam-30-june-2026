"""Script to compare two Briscola policies over 1000 games (supports learner as well)."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Add the project root directory to the sys path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation import play_match
from policy import (
    AdvancedHeuristicPolicy,
    GreedyPolicy,
    HeuristicPolicy,
    LinearSoftmaxPolicy,
    PerfectHeuristicPolicy,
    RandomPolicy,
    BriscolaFeatureExtractor,
)

# ==============================================================================
# CONFIGURATION - Modify these variables to choose the policies to compare!
# Available options:
#   - "perfect_heuristic"
#   - "advanced_heuristic"
#   - "heuristic"
#   - "greedy"
#   - "random"
#   - "learner" (requires specifying the relative CHECKPOINT_PATH below)

# python3 confronta_policy.py --policy_a learner --checkpoint_a experiments/results/nome_checkpoint.json --policy_b perfect_heuristic

# ==============================================================================
POLICY_A = "perfect_heuristic"
POLICY_B = "advanced_heuristic"

# If using the "learner" policy, specify the checkpoint path here:
CHECKPOINT_PATH_A = "experiments/results/checkpoint_perfect.json"
CHECKPOINT_PATH_B = "experiments/results/checkpoint_perfect.json"

NUMERO_PARTITE = 1000
MODALITA_GREEDY = False
SEED_INIZIALE = 1234
# ==============================================================================


def load_learner(checkpoint_path: str) -> LinearSoftmaxPolicy:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    checkpoint = json.loads(path.read_text(encoding="utf-8"))

    from policy.new_feature_set import NewFeatureSetExtractor
    
    saved_features = checkpoint.get("feature_names")
    
    # Try default BriscolaFeatureExtractor first
    extractor = BriscolaFeatureExtractor()
    if saved_features == list(extractor.feature_names):
        pass
    else:
        # Fallback to NewFeatureSetExtractor
        new_extractor = NewFeatureSetExtractor()
        if saved_features == list(new_extractor.feature_names):
            extractor = new_extractor
        else:
            raise ValueError(
                f"Checkpoint features ({path.name}) do not match any known extractor "
                "(BriscolaFeatureExtractor or NewFeatureSetExtractor)."
            )

    learner_data = checkpoint["learner"]
    return LinearSoftmaxPolicy(
        theta=learner_data["theta"],
        feature_extractor=extractor,
        name=learner_data.get("name", f"learner_{path.stem}"),
    )


def get_policy(name: str, checkpoint_path: str) -> any:
    name = name.lower().strip()
    if name == "perfect_heuristic":
        return PerfectHeuristicPolicy()
    elif name == "advanced_heuristic":
        return AdvancedHeuristicPolicy()
    elif name == "heuristic":
        return HeuristicPolicy()
    elif name == "greedy":
        return GreedyPolicy()
    elif name == "random":
        return RandomPolicy()
    elif name == "learner":
        return load_learner(checkpoint_path)
    else:
        raise ValueError(
            f"Unknown policy: '{name}'. Choose from: "
            "perfect_heuristic, advanced_heuristic, heuristic, greedy, random, learner"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two Briscola policies over a specified number of games."
    )
    parser.add_argument(
        "--policy_a",
        type=str,
        default=POLICY_A,
        help=f"The policy for Team A (players 0 and 2). Default: {POLICY_A}",
    )
    parser.add_argument(
        "--policy_b",
        type=str,
        default=POLICY_B,
        help=f"The policy for Team B (players 1 and 3). Default: {POLICY_B}",
    )
    parser.add_argument(
        "--checkpoint_a",
        type=str,
        default=CHECKPOINT_PATH_A,
        help=f"Checkpoint for policy A (if 'learner'). Default: {CHECKPOINT_PATH_A}",
    )
    parser.add_argument(
        "--checkpoint_b",
        type=str,
        default=CHECKPOINT_PATH_B,
        help=f"Checkpoint for policy B (if 'learner'). Default: {CHECKPOINT_PATH_B}",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=NUMERO_PARTITE,
        help=f"Number of games to play. Default: {NUMERO_PARTITE}",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        default=MODALITA_GREEDY,
        help=f"If specified, actions are selected greedily. Default: {MODALITA_GREEDY}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED_INIZIALE,
        help=f"Base seed for games. Default: {SEED_INIZIALE}",
    )
    args = parser.parse_args()

    try:
        policy_a = get_policy(args.policy_a, args.checkpoint_a)
        policy_b = get_policy(args.policy_b, args.checkpoint_b)
    except Exception as e:
        print(f"Error initializing policies: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Comparison in progress: {policy_a.name} (Team A) vs {policy_b.name} (Team B)")
    print(f"Number of games: {args.games}")
    print(f"Action selection mode: {'Greedy' if args.greedy else 'Stochastic'}")
    print(f"Initial seed: {args.seed}")
    print("-" * 50)

    wins_a = 0
    wins_b = 0
    pareggi = 0
    total_point_diff = 0.0

    policies_by_player = {
        0: policy_a,
        2: policy_a,
        1: policy_b,
        3: policy_b,
    }

    base_seed_ambiente = args.seed
    base_seed_policy = args.seed + 10000

    for i in range(args.games):
        primo_giocatore_id = i % 4
        res = play_match(
            policies_by_player=policies_by_player,
            seed_ambiente=base_seed_ambiente + i,
            seed_policy=base_seed_policy + i,
            primo_giocatore_id=primo_giocatore_id,
            greedy=args.greedy,
        )

        punti_a = res.punteggi_finali["pari"]
        punti_b = res.punteggi_finali["dispari"]

        total_point_diff += (punti_a - punti_b)

        if res.squadra_vincitrice == "pari":
            wins_a += 1
        elif res.squadra_vincitrice == "dispari":
            wins_b += 1
        else:
            pareggi += 1

    avg_point_diff = total_point_diff / args.games

    print("\n--- RESULTS ---")
    print(f"Won by Team A ({policy_a.name}): {wins_a} ({wins_a / args.games * 100:.1f}%)")
    print(f"Won by Team B ({policy_b.name}): {wins_b} ({wins_b / args.games * 100:.1f}%)")
    print(f"Draws: {pareggi} ({pareggi / args.games * 100:.1f}%)")
    print(f"Average point difference (A - B): {avg_point_diff:+.2f}")


if __name__ == "__main__":
    main()
