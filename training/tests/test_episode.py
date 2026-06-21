from __future__ import annotations

import random
import unittest
from dataclasses import dataclass, field

from game.cards import Carta
from game.observation import Osservazione
from training.episode import (
    MOSSE_PER_GIOCATORE,
    MOSSE_TOTALI_PARTITA,
    collect_episode,
)
from training.rewards import PUNTI_TOTALI_PARTITA, RewardConfig


@dataclass
class TrackingPolicy:
    name: str
    mode: str = "first"
    seen_players: list[int] = field(default_factory=list)
    greedy_values: list[bool] = field(default_factory=list)
    selected_cards: list[Carta] = field(default_factory=list)

    def action_probabilities(self, osservazione: Osservazione) -> dict[Carta, float]:
        probability = 1.0 / len(osservazione.azioni_legali)
        return {carta: probability for carta in osservazione.azioni_legali}

    def select_action(
        self,
        osservazione: Osservazione,
        rng: random.Random,
        greedy: bool = False,
    ) -> Carta:
        self.seen_players.append(osservazione.giocatore_id)
        self.greedy_values.append(greedy)
        if self.mode == "random":
            carta = rng.choice(list(osservazione.azioni_legali))
        else:
            carta = osservazione.azioni_legali[0]
        self.selected_cards.append(carta)
        return carta


class TestEpisodeCollection(unittest.TestCase):
    def test_collect_episode_produce_partita_completa(self):
        # Il collector deve produrre una traiettoria completa e coerente.
        policy = TrackingPolicy("shared")

        episode = collect_episode(
            learner_policy=policy,
            compagno_policy=policy,
            avversario_successivo_policy=policy,
            avversario_precedente_policy=policy,
            learner_giocatore_id=0,
            seed_ambiente=1,
            primo_giocatore_id=0,
            rng_policy=random.Random(2),
        )

        self.assertEqual(len(episode.rewards), MOSSE_TOTALI_PARTITA)
        self.assertEqual(len(episode.steps), MOSSE_PER_GIOCATORE)
        self.assertEqual(sum(episode.punteggi_finali.values()), PUNTI_TOTALI_PARTITA)
        self.assertAlmostEqual(episode.episode_return, sum(episode.rewards))
        self.assertEqual(episode.learner_giocatore_id, 0)
        self.assertEqual(episode.learner_squadra, "pari")

    def test_global_step_index_del_learner_usa_i_turni_globali(self):
        # Gli indici del learner devono riferirsi alle 40 mosse globali.
        policy = TrackingPolicy("shared")

        episode = collect_episode(
            learner_policy=policy,
            compagno_policy=policy,
            avversario_successivo_policy=policy,
            avversario_precedente_policy=policy,
            learner_giocatore_id=0,
            seed_ambiente=2,
            primo_giocatore_id=0,
            rng_policy=random.Random(3),
        )

        expected_indexes = [
            index
            for index, giocatore_id in enumerate(policy.seen_players)
            if giocatore_id == 0
        ]
        self.assertEqual(
            [step.global_step_index for step in episode.steps],
            expected_indexes,
        )

    def test_reward_to_go_usa_global_step_index(self):
        # Con reward solo terminale, ogni decisione vede la stessa reward futura.
        policy = TrackingPolicy("shared")

        episode = collect_episode(
            learner_policy=policy,
            compagno_policy=policy,
            avversario_successivo_policy=policy,
            avversario_precedente_policy=policy,
            learner_giocatore_id=0,
            seed_ambiente=3,
            primo_giocatore_id=0,
            rng_policy=random.Random(4),
            reward_config=RewardConfig(mode="combined_terminal"),
        )

        self.assertTrue(all(reward == 0.0 for reward in episode.rewards[:-1]))
        for step in episode.steps:
            self.assertAlmostEqual(step.reward_to_go, episode.episode_return)

    def test_dense_presa_aggiunge_reward_intermedie(self):
        # dense_presa integra reward immediate quando una presa viene completata.
        policy = TrackingPolicy("shared")

        episode = collect_episode(
            learner_policy=policy,
            compagno_policy=policy,
            avversario_successivo_policy=policy,
            avversario_precedente_policy=policy,
            learner_giocatore_id=0,
            seed_ambiente=4,
            primo_giocatore_id=0,
            rng_policy=random.Random(5),
            reward_config=RewardConfig(mode="dense_presa"),
        )

        self.assertTrue(any(reward != 0.0 for reward in episode.rewards[:-1]))

    def test_policy_mapping_usa_distanza_di_turno_dal_learner(self):
        # Le policy sono assegnate per successivo, compagno e precedente nel turno.
        learner = TrackingPolicy("learner")
        successivo = TrackingPolicy("successivo")
        compagno = TrackingPolicy("compagno")
        precedente = TrackingPolicy("precedente")

        collect_episode(
            learner_policy=learner,
            compagno_policy=compagno,
            avversario_successivo_policy=successivo,
            avversario_precedente_policy=precedente,
            learner_giocatore_id=1,
            seed_ambiente=5,
            primo_giocatore_id=0,
            rng_policy=random.Random(6),
        )

        self.assertEqual(set(learner.seen_players), {1})
        self.assertEqual(set(successivo.seen_players), {2})
        self.assertEqual(set(compagno.seen_players), {3})
        self.assertEqual(set(precedente.seen_players), {0})

    def test_stesso_seed_ambiente_e_rng_policy_rende_episode_riproducibile(self):
        # Ambiente e RNG policy separati restano riproducibili se reinizializzati uguali.
        first_learner = TrackingPolicy("learner", mode="random")
        second_learner = TrackingPolicy("learner", mode="random")

        first = collect_episode(
            learner_policy=first_learner,
            compagno_policy=TrackingPolicy("compagno", mode="random"),
            avversario_successivo_policy=TrackingPolicy("successivo", mode="random"),
            avversario_precedente_policy=TrackingPolicy("precedente", mode="random"),
            learner_giocatore_id=0,
            seed_ambiente=6,
            primo_giocatore_id=0,
            rng_policy=random.Random(7),
        )
        second = collect_episode(
            learner_policy=second_learner,
            compagno_policy=TrackingPolicy("compagno", mode="random"),
            avversario_successivo_policy=TrackingPolicy("successivo", mode="random"),
            avversario_precedente_policy=TrackingPolicy("precedente", mode="random"),
            learner_giocatore_id=0,
            seed_ambiente=6,
            primo_giocatore_id=0,
            rng_policy=random.Random(7),
        )

        self.assertEqual(first.punteggi_finali, second.punteggi_finali)
        self.assertEqual(first_learner.selected_cards, second_learner.selected_cards)
        self.assertEqual(first.episode_return, second.episode_return)

    def test_greedy_non_learner_non_modifica_il_learner(self):
        # Il learner resta stocastico; greedy_non_learner vale solo per gli altri.
        learner = TrackingPolicy("learner")
        successivo = TrackingPolicy("successivo")
        compagno = TrackingPolicy("compagno")
        precedente = TrackingPolicy("precedente")

        collect_episode(
            learner_policy=learner,
            compagno_policy=compagno,
            avversario_successivo_policy=successivo,
            avversario_precedente_policy=precedente,
            learner_giocatore_id=0,
            seed_ambiente=7,
            primo_giocatore_id=0,
            rng_policy=random.Random(8),
            greedy_non_learner=True,
        )

        self.assertEqual(set(learner.greedy_values), {False})
        self.assertEqual(set(successivo.greedy_values), {True})
        self.assertEqual(set(compagno.greedy_values), {True})
        self.assertEqual(set(precedente.greedy_values), {True})

    def test_giocatori_invalidi_solleva_value_error(self):
        # Gli id giocatore non validi vengono rifiutati prima di iniziare l'episodio.
        policy = TrackingPolicy("shared")

        with self.assertRaises(ValueError):
            collect_episode(
                learner_policy=policy,
                compagno_policy=policy,
                avversario_successivo_policy=policy,
                avversario_precedente_policy=policy,
                learner_giocatore_id=4,
                seed_ambiente=8,
                primo_giocatore_id=0,
                rng_policy=random.Random(9),
            )

        with self.assertRaises(ValueError):
            collect_episode(
                learner_policy=policy,
                compagno_policy=policy,
                avversario_successivo_policy=policy,
                avversario_precedente_policy=policy,
                learner_giocatore_id=0,
                seed_ambiente=8,
                primo_giocatore_id=4,
                rng_policy=random.Random(9),
            )


if __name__ == "__main__":
    unittest.main()
