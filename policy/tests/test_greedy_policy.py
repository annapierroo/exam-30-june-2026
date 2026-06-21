from __future__ import annotations

import random
import unittest

from game.cards import Carta, CartaGiocata
from game.observation import Osservazione
from policy import GreedyPolicy


def osservazione(
    *,
    mano: tuple[Carta, ...],
    carte_sul_campo: tuple[CartaGiocata, ...] = (),
    seme_briscola: str = "denari",
) -> Osservazione:
    return Osservazione(
        giocatore_id=0,
        compagno_id=2,
        avversario_sinistro_id=1,
        avversario_destro_id=3,
        mano=mano,
        mano_compagno_visibile=False,
        mano_compagno=(),
        seme_briscola=seme_briscola,
        briscola_esposta=Carta(seme_briscola, "asso"),
        proprietario_briscola_esposta=None,
        carte_sul_campo=carte_sul_campo,
        carte_giocate=carte_sul_campo,
        vincitori_prese=(),
        squadra="pari",
        squadra_avversaria="dispari",
        punteggio_squadra=0,
        punteggio_avversari=0,
        primo_giocatore_presa=0,
        giocatore_corrente=0,
        carte_nel_mazzo=28,
        indice_presa=0,
        posizione_nella_presa=len(carte_sul_campo),
    )


class TestGreedyPolicy(unittest.TestCase):
    def test_apre_con_carta_meno_costosa(self):
        carta_meno_costosa = Carta("coppe", "due")
        obs = osservazione(
            mano=(
                Carta("denari", "due"),
                Carta("coppe", "asso"),
                carta_meno_costosa,
            )
        )
        policy = GreedyPolicy()

        action = policy.select_action(obs, rng=random.Random(0))

        self.assertEqual(action, carta_meno_costosa)

    def test_prende_con_minima_sufficiente_dello_stesso_seme(self):
        minima_sufficiente = Carta("coppe", "sette")
        obs = osservazione(
            mano=(
                Carta("coppe", "fante"),
                minima_sufficiente,
                Carta("bastoni", "due"),
            ),
            carte_sul_campo=(
                CartaGiocata(giocatore_id=1, carta=Carta("coppe", "sei")),
            ),
        )
        policy = GreedyPolicy()

        action = policy.select_action(obs, rng=random.Random(0))

        self.assertEqual(action, minima_sufficiente)

    def test_prende_con_briscola_meno_costosa_quando_serve(self):
        briscola_meno_costosa = Carta("denari", "due")
        obs = osservazione(
            mano=(
                Carta("coppe", "due"),
                Carta("denari", "cinque"),
                briscola_meno_costosa,
            ),
            carte_sul_campo=(
                CartaGiocata(giocatore_id=1, carta=Carta("coppe", "asso")),
            ),
            seme_briscola="denari",
        )
        policy = GreedyPolicy()

        action = policy.select_action(obs, rng=random.Random(0))

        self.assertEqual(action, briscola_meno_costosa)

    def test_scarto_meno_costoso_quando_non_puo_prendere(self):
        scarto = Carta("coppe", "due")
        obs = osservazione(
            mano=(
                scarto,
                Carta("bastoni", "asso"),
                Carta("spade", "quattro"),
            ),
            carte_sul_campo=(
                CartaGiocata(giocatore_id=1, carta=Carta("denari", "asso")),
            ),
            seme_briscola="denari",
        )
        policy = GreedyPolicy()

        action = policy.select_action(obs, rng=random.Random(0))

        self.assertEqual(action, scarto)

    def test_probabilita_divise_tra_carte_migliori_equivalenti(self):
        prima = Carta("coppe", "due")
        seconda = Carta("bastoni", "due")
        obs = osservazione(mano=(prima, seconda, Carta("denari", "due")))
        policy = GreedyPolicy()

        probabilities = policy.action_probabilities(obs)

        self.assertAlmostEqual(probabilities[prima], 0.5)
        self.assertAlmostEqual(probabilities[seconda], 0.5)
        self.assertAlmostEqual(probabilities[Carta("denari", "due")], 0.0)
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)

    def test_mano_vuota_solleva_value_error(self):
        obs = osservazione(mano=())
        policy = GreedyPolicy()

        with self.assertRaises(ValueError):
            policy.action_probabilities(obs)

        with self.assertRaises(ValueError):
            policy.select_action(obs, rng=random.Random(0))


if __name__ == "__main__":
    unittest.main()
