import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert


# ==========================================================
# T3 - FM: índice de modulação, largura de banda e ruído
# Tópico 4 - Etapa 1: Validação sem ruído
# ==========================================================


# 1. PARÂMETROS DA SIMULAÇÃO

fs = 500_000       # Frequência de amostragem (Hz)
fc = 30_000        # Frequência da portadora (Hz)
fm = 1_000         # Frequência da mensagem (Hz)

T = 1.0            # Duração do sinal (s)
Ac = 1.0           # Amplitude da portadora

betas = [1, 5]     # Índices de modulação

t = np.arange(int(fs * T)) / fs


# 2. GERAÇÃO DA MENSAGEM ORIGINAL

m = np.cos(2 * np.pi * fm * t)


# 3. MODULADOR FM

def modular_fm(t, fc, fm, beta, Ac):

    fase = 2 * np.pi * fc * t
    fase += beta * np.sin(2 * np.pi * fm * t)

    return Ac * np.cos(fase)


# 4. DERIVADA ESPECTRAL

def derivar(sinal, fs):

    frequencias = np.fft.fftfreq(len(sinal), d=1/fs)
    espectro = np.fft.fft(sinal)

    derivada = np.fft.ifft(
        1j * 2 * np.pi * frequencias * espectro
    )

    return np.real(derivada)


# 5. DEMODULADOR FM

def demodular_fm(sinal, fs, fc, fm, beta, Ac):

    # Limitador de amplitude
    sinal_analitico = hilbert(sinal)

    amplitude = np.maximum(
        np.abs(sinal_analitico), 1e-12
    )

    sinal_limitado = Ac * np.real(
        sinal_analitico / amplitude
    )

    # Derivada do sinal
    derivada = derivar(sinal_limitado, fs)

    # Detector de envelope
    envelope = np.abs(hilbert(derivada))

    # Remoção da componente DC e correção do ganho
    mensagem_recuperada = (
        envelope / (2 * np.pi * Ac) - fc
    ) / (beta * fm)

    return mensagem_recuperada


# 6. SIMULAÇÃO E VISUALIZAÇÃO

fig, axs = plt.subplots(
    3, 2, figsize=(13, 8), sharex=True
)

# Mostrar apenas os primeiros 2 ms
janela = t < 0.002

for i, beta in enumerate(betas):

    # Modulação
    s_fm = modular_fm(t, fc, fm, beta, Ac)

    # Demodulação
    m_rec = demodular_fm(
        s_fm, fs, fc, fm, beta, Ac
    )

    # Erro de reconstrução
    erro = m_rec - m

    rmse = np.sqrt(np.mean(erro**2))

    # Previsões teóricas
    delta_f = beta * fm
    B_fm = 2 * (delta_f + fm)

    print(f"\nBeta = {beta}")
    print(f"Desvio de frequência: {delta_f:.0f} Hz")
    print(f"Banda de Carson: {B_fm:.0f} Hz")
    print(f"RMSE: {rmse:.6e}")

    # Mensagem original
    axs[0, i].plot(
        t[janela] * 1000,
        m[janela],
        color="black"
    )

    axs[0, i].set_title(
        f"Mensagem original - Beta = {beta}"
    )

    # Sinal FM
    axs[1, i].plot(
        t[janela] * 1000,
        s_fm[janela],
        color="darkcyan"
    )

    axs[1, i].set_title("Sinal modulado em FM")

    # Comparação entre mensagem original e recuperada
    axs[2, i].plot(
        t[janela] * 1000,
        m[janela],
        color="black",
        linewidth=2,
        label="Original"
    )

    axs[2, i].plot(
        t[janela] * 1000,
        m_rec[janela],
        color="darkorange",
        linestyle="--",
        label="Recuperada"
    )

    axs[2, i].set_title("Mensagem demodulada")
    axs[2, i].set_xlabel("Tempo (ms)")
    axs[2, i].legend()


for ax in axs.flat:
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)


fig.suptitle(
    "Validação da modulação e demodulação FM sem ruído",
    fontsize=14
)

plt.tight_layout()
plt.show()
