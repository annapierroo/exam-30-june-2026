"""Pure aggregate metrics for Briscola evaluation results."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean, stdev


@dataclass(frozen=True)
class EvaluationMetrics:
    """Aggregate metrics where one match is one statistical unit."""

    games: int
    win_rate: float
    draw_rate: float
    loss_rate: float
    mean_point_difference: float
    standard_error: float
    confidence_interval_95: tuple[float, float]


def compute_metrics(point_differences: list[int]) -> EvaluationMetrics:
    """Compute evaluation metrics from per-match point differences."""

    if not point_differences:
        raise ValueError("Serve almeno una partita per calcolare le metriche")

    games = len(point_differences)
    wins = sum(1 for difference in point_differences if difference > 0)
    draws = sum(1 for difference in point_differences if difference == 0)
    losses = sum(1 for difference in point_differences if difference < 0)
    average = float(mean(point_differences))
    stderr = standard_error(point_differences)
    confidence_interval = (
        average - 1.96 * stderr,
        average + 1.96 * stderr,
    )

    return EvaluationMetrics(
        games=games,
        win_rate=wins / games,
        draw_rate=draws / games,
        loss_rate=losses / games,
        mean_point_difference=average,
        standard_error=stderr,
        confidence_interval_95=confidence_interval,
    )


def standard_error(values: list[int]) -> float:
    """Estimate standard error over independent match-level values."""

    if not values:
        raise ValueError("Serve almeno un valore")
    if len(values) == 1:
        return 0.0
    return float(stdev(values) / math.sqrt(len(values)))
