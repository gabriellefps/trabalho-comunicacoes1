
import argparse
import csv
from pathlib import Path
from math import gcd

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, special, integrate
from scipy.io import wavfile


# ==========================================
# PARAMETROS DA SIMULACAO
# ==========================================

FS = 192000
FC = 40000
FM = 1000
B_AUDIO = 3000

BETAS = [0.3, 1, 5, 10]
BETAS_AUDIO = [0.3, 5]

DURACAO_AUDIO = 5
INICIO_AUDIO = 0


# ==========================================
# FUNCOES AUXILIARES
# ==========================================

def espectro(s):

    # FFT unilateral: amplitude e potencia
    X = np.fft.rfft(s) / len(s)
    f = np.fft.rfftfreq(len(s), 1 / FS)

    a = 2 * np.abs(X)
    p = 2 * np.abs(X)**2

    # Correcao da componente DC
    a[0] /= 2
    p[0] /= 2

    # Correcao da componente de Nyquist
    if len(s) % 2 == 0:
        a[-1] /= 2
        p[-1] /= 2

    return f, a, p


def banda99(f, p):

    # Calcula a potencia acumulada
    c = np.cumsum(p) / np.sum(p)

    # Exclui ate 0,5% de cada extremidade
    i, j = np.searchsorted(c, [0.005, 0.995])

    return f[j] - f[i], f[i], f[j]


def modular_tom(t, beta):

    return np.cos(
        2 * np.pi * FC * t
        + beta * np.sin(2 * np.pi * FM * t)
    )


def modular_audio(t, integral, beta):

    return np.cos(
        2 * np.pi * FC * t
        + 2 * np.pi * beta * B_AUDIO * integral
    )


def carregar_audio(caminho):

    fs, x = wavfile.read(caminho)

    # Converte para ponto flutuante
    x = x.astype(float)

    # Converte stereo para mono
    if x.ndim == 2:
        x = x.mean(axis=1)

    # Seleciona o trecho do audio
    inicio = int(INICIO_AUDIO * fs)
    fim = inicio + int(DURACAO_AUDIO * fs)

    x = x[inicio:fim]

    if len(x) < fs / 10 or fs <= 2 * B_AUDIO:
        raise ValueError(
            "Use WAV com pelo menos 0,1 s "
            "e taxa acima de 6000 Hz."
        )

    # Remove componente DC
    x -= x.mean()

    # Reamostragem para FS
    g = gcd(FS, fs)

    x = signal.resample_poly(
        x, FS // g, fs // g
    )

    # Filtro passa-baixas Butterworth
    filtro = signal.butter(
        6,
        B_AUDIO,
        fs=FS,
        output="sos"
    )

    x = signal.sosfiltfilt(filtro, x)

    # Remove componente DC novamente
    x -= x.mean()

    # Normalizacao da amplitude
    pico = np.max(np.abs(x))

    if not np.isfinite(pico) or pico < 1e-12:
        raise ValueError("Audio sem sinal util.")

    return x / pico


def trecho_energetico(m, duracao=0.1):

    # Seleciona a janela de maior energia
    k = min(len(m), int(duracao * FS))

    energia = np.concatenate(
        ([0.0], np.cumsum(m**2))
    )

    energia_janelas = energia[k:] - energia[:-k]

    inicio = int(np.argmax(energia_janelas))
    fim = inicio + k

    return inicio, fim


def salvar(fig, pasta, nome):

    fig.savefig(
        pasta / nome,
        dpi=200,
        bbox_inches="tight"
    )


def registrar(linhas, tipo, beta, B, bw, efetiva):

    desvio = beta * B

    carson = 2 * (desvio + B)

    alternativo = 2 * (desvio + efetiva)

    linhas.append([
        tipo,
        beta,
        desvio,
        B,
        efetiva,
        bw,
        carson,
        alternativo,
        100 * (bw - carson) / carson
    ])


# ==========================================
# PROGRAMA PRINCIPAL
# ==========================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--audio",
        type=Path,
        default=Path("audio_teste.wav")
    )

    parser.add_argument(
        "--sem-janelas",
        action="store_true"
    )

    args = parser.parse_args()

    pasta = Path("resultados")
    pasta.mkdir(exist_ok=True)

    linhas = []
    bessel = []

    # ======================================
    # 1. TOM PURO: FFT E BESSEL
    # ======================================

    t = np.arange(FS) / FS

    m = np.cos(2 * np.pi * FM * t)

    fig, axs = plt.subplots(
        len(BETAS),
        2,
        figsize=(13, 12),
        squeeze=False,
        constrained_layout=True
    )

    fig.suptitle(
        "Modulacao FM tonal: comparacao FFT e Bessel",
        fontsize=15
    )

    for i, beta in enumerate(BETAS):

        # Geracao do sinal FM
        s = modular_tom(t, beta)

        # Calculo da FFT
        f, a, p = espectro(s)

        # Ordens das componentes espectrais
        ordem = int(np.ceil(beta + 8))

        n = np.arange(-ordem, ordem + 1)

        fn = FC + n * FM

        indices = np.rint(
            fn / (f[1] - f[0])
        ).astype(int)

        # Amplitudes teoricas e experimentais
        teoria = np.abs(special.jv(n, beta))

        medido = a[indices]

        erro = np.max(
            np.abs(teoria - medido)
        )

        # Verificacao automatica
        assert erro < 1e-8, "FFT e Bessel divergiram"

        for ni, fi, at, am in zip(
            n, fn, teoria, medido
        ):
            bessel.append([
                beta,
                ni,
                fi,
                at,
                am,
                abs(at - am)
            ])

        # Grafico no dominio do tempo
        k = int(0.002 * FS)

        ax = axs[i, 0]

        ax.plot(
            t[:k] * 1000,
            s[:k],
            label="FM",
            linewidth=0.8
        )

        ax.plot(
            t[:k] * 1000,
            m[:k],
            "--",
            label="Mensagem",
            linewidth=1.5
        )

        ax.set(
            title=f"Tom: beta = {beta}",
            xlabel="Tempo (ms)",
            ylabel="Amplitude"
        )

        ax.grid(alpha=0.2)
        ax.legend(loc="upper right")

        # Comparacao FFT x Bessel
        ax = axs[i, 1]

        ax.stem(
            (fn - FC) / 1000,
            medido,
            basefmt=" ",
            label="FFT"
        )

        ax.scatter(
            (fn - FC) / 1000,
            teoria,
            facecolors="none",
            edgecolors="red",
            s=65,
            label="Bessel"
        )

        ax.set(
            title=f"Erro maximo: {erro:.2e}",
            xlabel="f - fc (kHz)",
            ylabel="Amplitude"
        )

        ax.grid(alpha=0.3)
        ax.legend()

    salvar(fig, pasta, "01_tom_bessel.png")


    # ======================================
    # 2. TOM PURO: REGRA DE CARSON
    # ======================================

    grade = sorted(set(
        np.round(
            np.arange(0, 10.01, 0.1), 2
        ).tolist() + BETAS
    ))

    bandas = []

    for beta in grade:

        s = modular_tom(t, beta)

        f, a, p = espectro(s)

        bw, f1, f2 = banda99(f, p)

        bandas.append(bw)

        registrar(
            linhas, "tom", beta, FM, bw, FM
        )

    fig, ax = plt.subplots(
        figsize=(10, 6),
        constrained_layout=True
    )

    ax.plot(
        grade,
        np.array(bandas) / 1000,
        label="FFT: 99%",
        linewidth=2
    )

    ax.plot(
        grade,
        2 * FM * (np.array(grade) + 1) / 1000,
        "--",
        label="Carson",
        linewidth=2
    )

    ax.set(
        title="Tom puro: largura de banda",
        xlabel="Indice de modulacao (beta)",
        ylabel="Largura de banda (kHz)"
    )

    ax.grid(alpha=0.4)
    ax.legend()

    salvar(fig, pasta, "02_carson_tom.png")


    # ======================================
    # 3. AUDIO: MODULACAO ESTREITA E LARGA
    # ======================================

    if args.audio.exists():

        m = carregar_audio(args.audio)

        t = np.arange(len(m)) / FS

        # Integral numerica da mensagem
        integral = integrate.cumulative_trapezoid(
            m, t, initial=0
        )

        # Largura de banda efetiva do audio
        f, a, p = espectro(m)

        B_ef = f[np.searchsorted(
            np.cumsum(p) / p.sum(), 0.99
        )]

        print("\nArquivo de audio:", args.audio)

        print(
            f"Duracao analisada: {len(m) / FS:.2f} s"
        )

        print(
            f"Banda efetiva do audio filtrado: "
            f"{B_ef:.1f} Hz"
        )

        # Seleciona os mesmos 100 ms
        # de maior energia para ambos os betas
        inicio, fim = trecho_energetico(m)

        tempo_ms = (
            t[inicio:fim] - t[inicio]
        ) * 1000

        print(
            f"Trecho visualizado: "
            f"{INICIO_AUDIO + t[inicio]:.3f} s a "
            f"{INICIO_AUDIO + t[fim - 1]:.3f} s"
        )

        fig, axs = plt.subplots(
            len(BETAS_AUDIO),
            2,
            figsize=(14, 9),
            squeeze=False,
            constrained_layout=True
        )

        fig.suptitle(
            f"Modulacao FM do audio: {args.audio.name}",
            fontsize=15
        )

        # Escala de frequencia comum aos graficos
        desvio_max = max(BETAS_AUDIO) * B_AUDIO

        for i, beta in enumerate(BETAS_AUDIO):

            # Gera o sinal FM
            s = modular_audio(
                t, integral, beta
            )

            # Calcula o espectro
            f, a, p = espectro(s)

            bw, f1, f2 = banda99(f, p)

            carson = 2 * B_AUDIO * (beta + 1)

            # Frequencia instantanea teorica
            fi = FC + beta * B_AUDIO * m

            # ==============================
            # GRAFICO TEMPORAL CORRIGIDO
            # ==============================

            ax = axs[i, 0]

            ax.plot(
                tempo_ms,
                fi[inicio:fim] / 1000,
                label="Frequencia instantanea",
                linewidth=1.2,
                color="tab:blue"
            )

            ax.axhline(
                FC / 1000,
                color="gray",
                linestyle="--",
                label="Portadora"
            )

            ax.set(
                title=f"Audio: beta = {beta}",
                xlabel="Tempo relativo (ms)",
                ylabel="Frequencia (kHz)"
            )

            # Mesma escala vertical para os dois betas
            ax.set_ylim(
                (FC - 1.05 * desvio_max) / 1000,
                (FC + 1.05 * desvio_max) / 1000
            )

            ax.grid(alpha=0.3)
            ax.legend(loc="upper right")

            # ==============================
            # ESPECTRO DO AUDIO MODULADO
            # ==============================

            ax = axs[i, 1]

            limite = max(
                carson / 2,
                abs(f1 - FC),
                abs(f2 - FC)
            ) * 1.5

            mask = np.abs(f - FC) <= limite

            ax.plot(
                (f[mask] - FC) / 1000,
                10 * np.log10(
                    np.maximum(
                        p[mask] / p.max(),
                        1e-15
                    )
                ),
                linewidth=0.8
            )

            # Limites previstos por Carson
            ax.axvline(
                -carson / 2000,
                ls="--",
                color="gray",
                label="Carson"
            )

            ax.axvline(
                carson / 2000,
                ls="--",
                color="gray"
            )

            # Limites medidos de 99%
            ax.axvline(
                (f1 - FC) / 1000,
                ls=":",
                color="red",
                label="Limites B99"
            )

            ax.axvline(
                (f2 - FC) / 1000,
                ls=":",
                color="red"
            )

            ax.set(
                title=f"B99 = {bw / 1000:.2f} kHz",
                xlabel="f - fc (kHz)",
                ylabel="Potencia relativa (dB)",
                ylim=(-85, 3)
            )

            ax.grid(alpha=0.2)
            ax.legend(loc="upper right")

        salvar(fig, pasta, "03_audio_fm.png")


        # ==================================
        # 4. AUDIO: REGRA DE CARSON
        # ==================================

        grade_audio = sorted(set(
            np.arange(
                0, 5.01, 0.25
            ).tolist() + BETAS_AUDIO
        ))

        bandas = []

        for beta in grade_audio:

            s = modular_audio(
                t, integral, beta
            )

            f, a, p = espectro(s)

            bw, f1, f2 = banda99(f, p)

            bandas.append(bw)

            registrar(
                linhas,
                "audio",
                beta,
                B_AUDIO,
                bw,
                B_ef
            )

        x = np.array(grade_audio)

        fig, ax = plt.subplots(
            figsize=(10, 6),
            constrained_layout=True
        )

        ax.plot(
            x,
            np.array(bandas) / 1000,
            "o-",
            label="FFT: 99%",
            linewidth=2
        )

        ax.plot(
            x,
            2 * B_AUDIO * (x + 1) / 1000,
            "--",
            label="Carson: banda nominal"
        )

        ax.plot(
            x,
            2 * (x * B_AUDIO + B_ef) / 1000,
            ":",
            label="Carson: banda efetiva",
            linewidth=2
        )

        ax.set(
            title="Audio: largura de banda",
            xlabel="Beta (referencia nominal)",
            ylabel="Largura de banda (kHz)"
        )

        ax.legend()
        ax.grid(alpha=0.4)

        salvar(fig, pasta, "04_carson_audio.png")


        # ==================================
        # 5. AUDIO ORIGINAL: GRAFICO EXTRA
        # ==================================

        fig, axs = plt.subplots(
            2,
            1,
            figsize=(12, 7),
            constrained_layout=True
        )

        fig.suptitle(
            f"Mensagem de audio: {args.audio.name}",
            fontsize=15
        )

        # Forma de onda dos 5 segundos
        axs[0].plot(
            t,
            m,
            linewidth=0.7
        )

        axs[0].axvspan(
            t[inicio],
            t[fim - 1],
            color="orange",
            alpha=0.3,
            label="Trecho visualizado na Figure 3"
        )

        axs[0].set(
            title="Audio filtrado e normalizado",
            xlabel="Tempo (s)",
            ylabel="Amplitude"
        )

        axs[0].grid(alpha=0.3)
        axs[0].legend()

        # Detalhe dos 100 ms selecionados
        axs[1].plot(
            tempo_ms,
            m[inicio:fim],
            color="tab:orange",
            linewidth=1.2
        )

        axs[1].set(
            title="Trecho de 100 ms com maior energia",
            xlabel="Tempo relativo (ms)",
            ylabel="Amplitude"
        )

        axs[1].set_ylim(-1.05, 1.05)
        axs[1].grid(alpha=0.3)

        salvar(fig, pasta, "05_audio_original.png")

    else:

        print(
            f"Audio nao encontrado: {args.audio}. "
            "Apenas o tom foi simulado."
        )


    # ======================================
    # 6. EXPORTACAO DAS TABELAS CSV
    # ======================================

    for nome, cab, dados in [

        (
            "largura_banda.csv",
            [
                "sinal",
                "beta",
                "desvio_Hz",
                "B_nominal_Hz",
                "B_efetiva_Hz",
                "B99_Hz",
                "Carson_Hz",
                "Carson_efetiva_Hz",
                "erro_percentual"
            ],
            linhas
        ),

        (
            "bessel.csv",
            [
                "beta",
                "ordem",
                "frequencia_Hz",
                "amplitude_teorica",
                "amplitude_fft",
                "erro_absoluto"
            ],
            bessel
        )

    ]:

        with (pasta / nome).open(
            "w",
            newline="",
            encoding="utf-8"
        ) as arq:

            w = csv.writer(arq)

            w.writerow(cab)
            w.writerows(dados)


    # ======================================
    # 7. RESULTADOS NO TERMINAL
    # ======================================

    print("\n" + "=" * 65)
    print("RESULTADOS DA SIMULACAO FM")
    print("=" * 65)

    for r in linhas:

        if (
            (r[0] == "tom" and r[1] in BETAS)
            or
            (r[0] == "audio" and r[1] in BETAS_AUDIO)
        ):

            print(
                f"{r[0]:5s} | "
                f"beta={r[1]:4g} | "
                f"B99={r[5] / 1000:6.2f} kHz | "
                f"Carson={r[6] / 1000:6.2f} kHz"
            )

    print("=" * 65)

    print(
        "\nResultados salvos em:",
        pasta.resolve()
    )

    if not args.sem_janelas:
        plt.show()

    plt.close("all")


if __name__ == "__main__":
    main()

