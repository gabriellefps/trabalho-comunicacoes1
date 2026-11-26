import numpy as np
from scipy import signal


# ==========================================
# TOPICO 3: DEMODULADOR DISCRIMINADOR
# ==========================================

def derivar_sinal(sinal_recebido, frequencia_amostragem):

    # Derivar no tempo equivale a multiplicar a FFT por j * 2 * pi * f
    frequencias = np.fft.rfftfreq(
        len(sinal_recebido), 1 / frequencia_amostragem
    )

    espectro = np.fft.rfft(sinal_recebido)

    derivada = np.fft.irfft(
        1j * 2 * np.pi * frequencias * espectro,
        n=len(sinal_recebido)
    )

    return derivada


def demodular_discriminador(
    sinal_recebido,
    frequencia_amostragem,
    frequencia_portadora,
    desvio_frequencia,
    banda_mensagem
):

    # ======================================
    # 1. VERIFICACAO DOS PARAMETROS
    # ======================================

    if np.iscomplexobj(sinal_recebido):
        raise ValueError("O discriminador recebe o sinal FM real.")

    sinal_recebido = np.asarray(sinal_recebido, dtype=float)

    if (
        sinal_recebido.ndim != 1
        or len(sinal_recebido) < 32
        or not np.all(np.isfinite(sinal_recebido))
    ):
        raise ValueError("Use um sinal real, finito e com pelo menos 32 amostras.")

    parametros = [
        frequencia_amostragem,
        frequencia_portadora,
        desvio_frequencia,
        banda_mensagem
    ]

    if not np.all(np.isfinite(parametros)) or min(parametros) <= 0:
        raise ValueError("As frequencias devem ser positivas e finitas.")

    if frequencia_portadora <= desvio_frequencia:
        raise ValueError("A portadora deve ser maior que o desvio de frequencia.")

    if (
        frequencia_portadora + desvio_frequencia + banda_mensagem
        >= frequencia_amostragem / 2
    ):
        raise ValueError("A banda de Carson deve ficar abaixo de Nyquist.")

    # ======================================
    # 2. LIMITADOR DE AMPLITUDE
    # ======================================

    envelope_entrada = np.abs(signal.hilbert(sinal_recebido))

    sinal_limitado = sinal_recebido / np.maximum(
        envelope_entrada, 1e-12
    )

    # ======================================
    # 3. DERIVADA DO SINAL FM
    # ======================================

    derivada = derivar_sinal(
        sinal_limitado, frequencia_amostragem
    )

    # ======================================
    # 4. DETECTOR DE ENVELOPE
    # ======================================

    envelope_derivada = np.abs(signal.hilbert(derivada))

    # Com amplitude unitaria, envelope = 2 * pi * (fc + desvio * mensagem)
    mensagem_recuperada = (
        envelope_derivada / (2 * np.pi) - frequencia_portadora
    ) / desvio_frequencia

    # ======================================
    # 5. FILTRO PASSA-BAIXAS IDEAL
    # ======================================

    frequencias = np.fft.rfftfreq(
        len(mensagem_recuperada), 1 / frequencia_amostragem
    )

    espectro_mensagem = np.fft.rfft(mensagem_recuperada)

    espectro_mensagem[frequencias > banda_mensagem] = 0

    return np.fft.irfft(
        espectro_mensagem,
        n=len(mensagem_recuperada)
    )


# ==========================================
# COMPARACAO COM A MENSAGEM ORIGINAL
# ==========================================

def medir_erro(mensagem_original, mensagem_recuperada):

    # Desconsidera 5% de cada borda por causa dos transitorios
    margem = max(1, int(0.05 * len(mensagem_original)))

    referencia = mensagem_original[margem:-margem]
    recuperada = mensagem_recuperada[margem:-margem]

    erro = recuperada - referencia

    raiz_erro_quadratico_medio = np.sqrt(np.mean(erro**2))
    amplitude_eficaz = np.sqrt(np.mean(referencia**2))

    erro_relativo = 100 * raiz_erro_quadratico_medio / max(
        amplitude_eficaz, 1e-12
    )

    return raiz_erro_quadratico_medio, erro_relativo


def comparar_mensagens(
    tempo_ms, mensagem_original, mensagem_recuperada, eixos
):

    eixos[0].plot(
        tempo_ms,
        mensagem_original,
        label="Original",
        color="black",
        linewidth=1.5
    )

    eixos[0].plot(
        tempo_ms,
        mensagem_recuperada,
        "--",
        label="Recuperada",
        color="tab:orange"
    )

    eixos[0].set_ylabel("Amplitude")
    eixos[0].legend(loc="upper right")

    eixos[1].plot(
        tempo_ms,
        mensagem_recuperada - mensagem_original,
        color="tab:red",
        linewidth=0.8
    )

    eixos[1].set_ylabel("Erro de reconstrucao")

    for eixo in eixos:
        eixo.set_xlabel("Tempo relativo (ms)")
        eixo.grid(alpha=0.3)
