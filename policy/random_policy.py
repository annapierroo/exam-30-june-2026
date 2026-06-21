"""Random baseline policy."""

from __future__ import annotations

import random
from dataclasses import dataclass

from game.cards import Carta
from game.observation import Osservazione


@dataclass
class RandomPolicy:
    """Policy that samples uniformly from legal actions."""

    name: str = "random"

    def action_probabilities(self, osservazione: Osservazione) -> dict[Carta, float]:
        azioni_legali = osservazione.azioni_legali
        if not azioni_legali:
            raise ValueError("No legal actions available")
        probability = 1.0 / len(azioni_legali)
        return {carta: probability for carta in azioni_legali}

    def select_action(
        self,
        osservazione: Osservazione,
        rng: random.Random,
        greedy: bool = False,
    ) -> Carta:
        azioni_legali = osservazione.azioni_legali
        if not azioni_legali:
            raise ValueError("No legal actions available")
        return rng.choice(list(azioni_legali))
