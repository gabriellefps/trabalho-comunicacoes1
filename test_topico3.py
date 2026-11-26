import unittest

import numpy as np

from topico3 import demodular_discriminador, derivar_sinal, medir_erro


# ==========================================
# VERIFICACAO DO DEMODULADOR SEM RUIDO
# ==========================================

class TesteDiscriminador(unittest.TestCase):

    def setUp(self):

        self.frequencia_amostragem = 192000
        self.frequencia_portadora = 40000
        self.frequencia_mensagem = 1000

        self.tempo = np.arange(19200) / self.frequencia_amostragem

        self.mensagem_original = np.cos(
            2 * np.pi * self.frequencia_mensagem * self.tempo
        )

    def test_derivada_de_senoide(self):

        derivada = derivar_sinal(
            self.mensagem_original, self.frequencia_amostragem
        )

        derivada_teorica = -2 * np.pi * self.frequencia_mensagem * np.sin(
            2 * np.pi * self.frequencia_mensagem * self.tempo
        )

        np.testing.assert_allclose(
            derivada, derivada_teorica, atol=1e-6, rtol=0
        )

    def test_tom_estreito_e_largo_com_amplitude_variavel(self):

        for beta in [0.3, 1, 5, 10]:
            for amplitude in [0.4, 1.0, 2.0]:
                with self.subTest(beta=beta, amplitude=amplitude):

                    sinal_fm = amplitude * np.cos(
                        2 * np.pi * self.frequencia_portadora * self.tempo
                        + beta * np.sin(
                            2 * np.pi * self.frequencia_mensagem * self.tempo
                        )
                    )

                    recuperada = demodular_discriminador(
                        sinal_fm,
                        self.frequencia_amostragem,
                        self.frequencia_portadora,
                        beta * self.frequencia_mensagem,
                        self.frequencia_mensagem
                    )

                    rmse, erro_relativo = medir_erro(
                        self.mensagem_original, recuperada
                    )

                    self.assertLess(rmse, 1e-6)
                    self.assertLess(erro_relativo, 0.001)

    def test_mensagem_com_duas_frequencias(self):

        mensagem = (
            0.6 * np.cos(2 * np.pi * 500 * self.tempo)
            + 0.3 * np.sin(2 * np.pi * 1500 * self.tempo)
        )

        integral = (
            0.6 * np.sin(2 * np.pi * 500 * self.tempo) / (2 * np.pi * 500)
            - 0.3 * np.cos(2 * np.pi * 1500 * self.tempo) / (2 * np.pi * 1500)
        )

        for desvio in [900, 15000]:
            with self.subTest(desvio=desvio):

                sinal_fm = np.cos(
                    2 * np.pi * self.frequencia_portadora * self.tempo
                    + 2 * np.pi * desvio * integral
                )

                recuperada = demodular_discriminador(
                    sinal_fm,
                    self.frequencia_amostragem,
                    self.frequencia_portadora,
                    desvio,
                    1500
                )

                rmse, _ = medir_erro(mensagem, recuperada)

                self.assertLess(rmse, 1e-6)

    def test_parametros_incompativeis(self):

        for desvio in [0, 40000, 60000]:
            with self.subTest(desvio=desvio):
                with self.assertRaises(ValueError):
                    demodular_discriminador(
                        self.mensagem_original,
                        self.frequencia_amostragem,
                        self.frequencia_portadora,
                        desvio,
                        self.frequencia_mensagem
                    )

        with self.assertRaises(ValueError):
            demodular_discriminador(
                self.mensagem_original, 80000, 30000, 15000, 3000
            )


if __name__ == "__main__":
    unittest.main()
