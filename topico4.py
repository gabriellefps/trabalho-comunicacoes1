"""Tópico 4: ruído em FM. Execute com --sem-janelas para apenas salvar.

O CSV preserva a previsão corrigida em toda a grade original; sua validade
acima do limiar é indicada nos gráficos. Importar este módulo não simula.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.signal import hilbert

# Parâmetros originais (Hz, segundos e amplitudes).
fs = 500_000
fc = 30_000
fm = 1_000
T = 1.0
Ac = 1.0
betas = [1, 5]
snr_desejada = 15

def modular_fm(t, fc, fm, beta, Ac):

    fase = 2 * np.pi * fc * t
    fase += beta * np.sin(2 * np.pi * fm * t)

    return Ac * np.cos(fase)

def derivar(sinal, fs):

    frequencias = np.fft.fftfreq(len(sinal), d=1/fs)
    espectro = np.fft.fft(sinal)

    derivada = np.fft.ifft(
        1j * 2 * np.pi * frequencias * espectro
    )

    return np.real(derivada)

def demodular_fm(sinal, fs, fc, fm, beta, Ac):

    # Limitador de amplitude
    sinal_analitico = np.asarray(hilbert(sinal))

    amplitude = np.maximum(
        np.abs(sinal_analitico), 1e-12
    )

    sinal_limitado = Ac * np.real(
        sinal_analitico / amplitude
    )

    # Derivada do sinal
    derivada = derivar(sinal_limitado, fs)

    # Detector de envelope
    envelope = np.abs(np.asarray(hilbert(derivada)))

    # Remoção da componente DC e correção do ganho
    mensagem_recuperada = (
        envelope / (2 * np.pi * Ac) - fc
    ) / (beta * fm)

    return mensagem_recuperada

def filtro_passa_faixa(sinal, fs, fc, largura_banda):

    frequencias = np.fft.fftfreq(len(sinal), d=1/fs)

    espectro = np.fft.fft(sinal)

    mascara = (
        np.abs(np.abs(frequencias) - fc)
        <= largura_banda / 2
    )

    return np.real(np.fft.ifft(espectro * mascara))

def adicionar_awgn(rng, sinal, fs, fc, largura_banda, snr_db):

    # Geracao de ruido branco gaussiano
    ruido = rng.normal(0, 1, len(sinal))

    # Filtragem na banda do receptor
    sinal_filtrado = filtro_passa_faixa(
        sinal, fs, fc, largura_banda
    )

    ruido_filtrado = filtro_passa_faixa(
        ruido, fs, fc, largura_banda
    )

    # Potencia do sinal
    Ps = np.mean(sinal_filtrado**2)

    # Potencia desejada do ruido
    Pn = Ps / (10**(snr_db / 10))

    # Ajuste da intensidade do ruido
    ruido_filtrado *= np.sqrt(
        Pn / np.mean(ruido_filtrado**2)
    )

    # Sinal recebido apos o filtro
    sinal_recebido = sinal_filtrado + ruido_filtrado

    # SNR efetivamente obtida
    snr_real = 10 * np.log10(
        Ps / np.mean(ruido_filtrado**2)
    )

    return sinal_recebido, snr_real

def filtro_passa_baixas(sinal, fs, frequencia_corte):

    frequencias = np.fft.fftfreq(len(sinal), d=1/fs)

    espectro = np.fft.fft(sinal)

    mascara = np.abs(frequencias) <= frequencia_corte

    sinal_filtrado = np.real(
        np.fft.ifft(espectro * mascara)
    )

    return sinal_filtrado

def calcular_snr(mensagem_original, mensagem_recuperada):

    erro = mensagem_recuperada - mensagem_original

    potencia_sinal = np.mean(mensagem_original**2)
    potencia_erro = np.mean(erro**2)

    return 10 * np.log10(
        potencia_sinal / max(potencia_erro, 1e-30)
    )

def recuperar_mensagem(sinal, beta):
    mensagem = demodular_fm(sinal, fs, fc, fm, beta, Ac)
    mensagem = filtro_passa_baixas(mensagem, fs, fm)
    mensagem -= np.mean(mensagem)
    return mensagem

def validar_sem_ruido(plt, t, m):
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
            color="#0072B2"
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
            color="#D55E00",
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
    return fig

def demonstrar_awgn(plt, rng, t):
    janela = t < 0.002
    # Visualizacao dos sinais com ruido
    fig2, axs2 = plt.subplots(
        2, 2, figsize=(13, 6), sharex=True
    )
    for i, beta in enumerate(betas):
        s_fm = modular_fm(t, fc, fm, beta, Ac)
        B_fm = 2 * fm * (beta + 1)
        s_ruidoso, snr_real = adicionar_awgn(
            rng, s_fm, fs, fc, B_fm, snr_desejada
        )
        print(f"\nSimulacao com ruido - Beta = {beta}")
        print(f"SNR desejada: {snr_desejada:.2f} dB")
        print(f"SNR obtida: {snr_real:.2f} dB")
        # Sinal FM original
        axs2[0, i].plot(
            t[janela] * 1000,
            s_fm[janela],
            color="#0072B2"
        )
        axs2[0, i].set_title(
            f"FM sem ruido - Beta = {beta}"
        )
        # Sinal FM com ruido
        axs2[1, i].plot(
            t[janela] * 1000,
            s_ruidoso[janela],
            color="#D55E00"
        )
        axs2[1, i].set_title(
            f"FM com ruido - SNR = {snr_real:.0f} dB"
        )
        axs2[1, i].set_xlabel("Tempo (ms)")
    for ax in axs2.flat:
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
    fig2.suptitle(
        "Comportamento do sinal FM na presenca de ruido AWGN"
    )
    fig2.tight_layout()
    return fig2

def demonstrar_demodulacao(plt, rng, t, m):
    # Comparacao entre mensagem original e recuperada
    fig3, axs3 = plt.subplots(
        2, 2, figsize=(13, 7), sharex=True
    )
    # Exibir os primeiros 5 ms
    janela_ruido = t < 0.005
    for i, beta in enumerate(betas):
        # Modulacao FM
        s_fm = modular_fm(t, fc, fm, beta, Ac)
        B_fm = 2 * fm * (beta + 1)
        # Canal com ruido AWGN
        s_ruidoso, snr_in = adicionar_awgn(
            rng, s_fm, fs, fc, B_fm, snr_desejada
        )
        # Demodulacao do sinal com ruido
        m_rec_ruido = recuperar_mensagem(s_ruidoso, beta)
        # Calculo da SNR de saida
        snr_out = calcular_snr(m, m_rec_ruido)
        # Ganho experimental
        ganho_exp = snr_out - snr_in
        # Ganho teorico
        ganho_teorico = 10 * np.log10(
            3 * beta**2 * (beta + 1)
        )
        snr_out_teorica = snr_in + ganho_teorico
        # Verificacao da distorcao do filtro sem ruido
        s_limpo_filtrado = filtro_passa_faixa(
            s_fm, fs, fc, B_fm
        )
        m_limpa = recuperar_mensagem(s_limpo_filtrado, beta)
        snr_sem_ruido = calcular_snr(m, m_limpa)
        # Resultados numericos
        print(f"\nDemodulacao com ruido - Beta = {beta}")
        print(f"SNR de entrada: {snr_in:.2f} dB")
        print(f"SNR de saida: {snr_out:.2f} dB")
        print(f"SNR de saida teorica: {snr_out_teorica:.2f} dB")
        print(f"Ganho experimental: {ganho_exp:.2f} dB")
        print(f"Ganho teorico: {ganho_teorico:.2f} dB")
        print(f"SNR sem ruido com filtros: {snr_sem_ruido:.2f} dB")
        # Comparacao entre as mensagens
        axs3[0, i].plot(
            t[janela_ruido] * 1000,
            m[janela_ruido],
            color="black",
            linewidth=2,
            label="Original"
        )
        axs3[0, i].plot(
            t[janela_ruido] * 1000,
            m_rec_ruido[janela_ruido],
            color="#D55E00",
            linestyle="--",
            label="Recuperada"
        )
        axs3[0, i].set_title(
            f"Beta = {beta} | SNR de saida = {snr_out:.2f} dB"
        )
        axs3[0, i].legend()
        # Erro de reconstrucao
        erro = m_rec_ruido - m
        axs3[1, i].plot(
            t[janela_ruido] * 1000,
            erro[janela_ruido],
            color="#0072B2"
        )
        axs3[1, i].set_title("Erro de reconstrucao")
        axs3[1, i].set_xlabel("Tempo (ms)")
    for ax in axs3.flat:
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
    # Mesma escala vertical para comparar os erros
    limite = max(
        abs(axs3[1, 0].get_ylim()[0]),
        abs(axs3[1, 0].get_ylim()[1]),
        abs(axs3[1, 1].get_ylim()[0]),
        abs(axs3[1, 1].get_ylim()[1])
    )
    for ax in axs3[1, :]:
        ax.set_ylim(-limite, limite)
    fig3.suptitle(
        "Demodulacao FM na presenca de ruido AWGN"
    )
    fig3.tight_layout()
    return fig3

def varrer_snr(rng):
    snr_entrada_db = np.arange(-10, 36, 1)
    num_realizacoes = 5
    T_varredura = 0.2
    t_varredura = np.arange(int(fs * T_varredura)) / fs
    m_varredura = np.cos(
        2 * np.pi * fm * t_varredura
    )
    potencia_mensagem = np.mean(m_varredura**2)
    resultados_snr = {}
    for i, beta in enumerate(betas):
        print(f"\nIniciando varredura para Beta = {beta}")
        B_fm = 2 * fm * (beta + 1)
        # Geracao do sinal FM
        s_fm = modular_fm(
            t_varredura, fc, fm, beta, Ac
        )
        snr_saida_db = []
        # Varredura da SNR de entrada
        for snr_in in snr_entrada_db:
            potencias_erro = []
            # Repeticoes com diferentes realizacoes de ruido
            for _ in range(num_realizacoes):
                s_ruidoso, _ = adicionar_awgn(
                    rng, s_fm, fs, fc, B_fm, snr_in
                )
                # Demodulacao FM
                m_rec = recuperar_mensagem(s_ruidoso, beta)
                # Potencia do erro de reconstrucao
                erro = m_rec - m_varredura
                potencias_erro.append(
                    np.mean(erro**2)
                )
            # Media das potencias de erro
            potencia_erro_media = np.mean(potencias_erro)
            snr_out = 10 * np.log10(
                potencia_mensagem / potencia_erro_media
            )
            snr_saida_db.append(snr_out)
            print(
                f"Beta = {beta} | "
                f"Entrada = {snr_in:3.0f} dB | "
                f"Saida = {snr_out:6.2f} dB"
            )
        snr_saida_db = np.asarray(snr_saida_db)
        # Previsao teorica acima do limiar
        ganho_teorico = 10 * np.log10(
            3 * beta**2 * (beta + 1)
        )
        snr_teorica_db = snr_entrada_db + ganho_teorico
        # Armazenamento dos resultados
        resultados_snr[beta] = {
            "entrada": snr_entrada_db,
            "saida": snr_saida_db,
            "teoria": snr_teorica_db
        }
    print("\nVarredura de SNR concluida!")
    return resultados_snr, t_varredura, m_varredura

def analisar_limiar(resultados_snr, t_varredura, m_varredura):
    tolerancia_db = 1.0
    pontos_consecutivos = 3
    print("\n" + "=" * 60)
    print("ANALISE FINAL DOS RESULTADOS")
    print("=" * 60)
    for beta in betas:
        entrada = resultados_snr[beta]["entrada"]
        saida = resultados_snr[beta]["saida"]
        teoria = resultados_snr[beta]["teoria"]
        # Diferenca entre teoria e simulacao
        diferenca = np.abs(teoria - saida)
        # Identificacao do efeito limiar
        limiar = None
        for j in range(len(entrada) - pontos_consecutivos + 1):
            trecho = diferenca[j:j + pontos_consecutivos]
            if np.all(trecho <= tolerancia_db):
                limiar = entrada[j]
                break
        resultados_snr[beta]["limiar"] = limiar
        # Largura de banda de Carson
        B_fm = 2 * fm * (beta + 1)
        # Medicao da distorcao sem ruido
        s_limpo = modular_fm(
            t_varredura, fc, fm, beta, Ac
        )
        s_limpo = filtro_passa_faixa(
            s_limpo, fs, fc, B_fm
        )
        m_limpa = recuperar_mensagem(s_limpo, beta)
        snr_limite = calcular_snr(
            m_varredura, m_limpa
        )
        # Previsao corrigida considerando a distorcao
        teoria_linear = 10**(teoria / 10)
        limite_linear = 10**(snr_limite / 10)
        teoria_corrigida = 10 * np.log10(
            1 / (1 / teoria_linear + 1 / limite_linear)
        )
        resultados_snr[beta]["teoria_corrigida"] = teoria_corrigida
        # Apresentacao dos resultados
        print(f"\nBeta = {beta}")
        if limiar is not None:
            print(f"Efeito limiar: {limiar:.2f} dB")
        else:
            print("Limiar nao identificado pelo criterio")
        print(f"SNR limite sem ruido: {snr_limite:.2f} dB")
        print(f"\nComparacao para entrada de {entrada[-1]} dB:")
        print(f"SNR teorica ideal: {teoria[-1]:.2f} dB")
        print(f"SNR experimental: {saida[-1]:.2f} dB")
        print(
            f"SNR teorica corrigida: "
            f"{teoria_corrigida[-1]:.2f} dB"
        )
        print(
            f"Erro da previsao corrigida: "
            f"{abs(teoria_corrigida[-1] - saida[-1]):.2f} dB"
        )
    return resultados_snr

def salvar_resultados(figuras, resultados_snr, pasta_resultados):
    # Salvamento dos graficos
    graficos = [
        (figuras[0], "06_fm_validacao.png"),
        (figuras[1], "07_fm_awgn.png"),
        (figuras[2], "08_fm_demodulacao_ruido.png"),
        (figuras[3], "09_fm_curvas_snr.png"),
        (figuras[4], "10_fm_teoria_corrigida.png")
    ]
    for figura, nome in graficos:
        figura.savefig(
            pasta_resultados / nome,
            dpi=300,
            bbox_inches="tight"
        )
        print(f"Grafico salvo: {nome}")
    # Salvamento dos resultados numericos em CSV
    for beta in betas:
        dados = resultados_snr[beta]
        tabela = np.column_stack((
            dados["entrada"],
            dados["saida"],
            dados["teoria"],
            dados["teoria_corrigida"]
        ))
        nome_csv = f"snr_beta_{beta}.csv"
        np.savetxt(
            pasta_resultados / nome_csv,
            tabela,
            delimiter=";",
            fmt="%.6f",
            header=(
                "SNR_entrada_dB;"
                "SNR_experimental_dB;"
                "SNR_teorica_dB;"
                "SNR_teorica_corrigida_dB"
            ),
            comments=""
        )
        print(f"Tabela salva: {nome_csv}")
    print("\nTodos os resultados foram salvos!")
    return None

def desenhar_teoria(ax, dados, cor, sufixo=""):
    """Diferencia a extrapolação sem mudar os valores calculados."""
    entrada, teoria = dados["entrada"], dados["teoria"]
    limiar = dados["limiar"]
    if limiar is None:
        ax.plot(entrada, teoria, color=cor, linestyle=":",
                label="Teoria ideal: limiar não identificado" + sufixo)
        return
    abaixo = entrada <= limiar  # Inclui o ponto de união entre os segmentos.
    acima = entrada >= limiar
    ax.plot(entrada[abaixo], teoria[abaixo], color=cor, linestyle=":",
            linewidth=1.5, label="Extrapolação abaixo do limiar" + sufixo)
    ax.plot(entrada[acima], teoria[acima], color=cor, linestyle="--",
            linewidth=1.5, label="Teoria ideal acima do limiar" + sufixo)
    ax.axvline(limiar, color=cor, linestyle=(0, (5, 2, 1, 2)),
               linewidth=1.2, label=f"Limiar: {limiar:g} dB" + sufixo)

def graficos_snr(plt, resultados_snr):
    fig4, ax4 = plt.subplots(figsize=(12, 8))
    fig5, axs5 = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)
    for i, (beta, cor, marcador) in enumerate(zip(betas, ["#0072B2", "#D55E00"], ["o", "s"])):
        dados = resultados_snr[beta]
        entrada, saida = dados["entrada"], dados["saida"]
        sufixo = f" — Beta = {beta}"
        ax4.plot(entrada, saida, color=cor, marker=marcador, markersize=3,
                 linewidth=2, label="Experimental" + sufixo)
        desenhar_teoria(ax4, dados, cor, sufixo)
        ax = axs5[i]
        ax.plot(entrada, saida, color="#0072B2", marker="o", markersize=3,
                label="Experimental")
        desenhar_teoria(ax, dados, "#666666")
        if dados["limiar"] is not None:
            regiao = entrada >= dados["limiar"]
            ax.plot(entrada[regiao], dados["teoria_corrigida"][regiao],
                    color="#D55E00", linestyle="-.", linewidth=2,
                    label="Teoria corrigida acima do limiar")
        ax.set_title(f"Beta = {beta}")
    for ax in [ax4, *axs5]:
        ax.set_xlabel("SNR de entrada (dB)")
        ax.set_ylabel("SNR de saída (dB)")
        ax.set_xticks(np.arange(-10, 36, 5))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    ax4.set_title("SNR de saída × SNR de entrada — Modulação FM")
    fig5.suptitle("Influência da distorção na SNR de saída da FM")
    for fig in [fig4, fig5]:
        fig.text(0.5, 0.015, "Limiar calculado: diferença absoluta ≤ 1 dB em 3 pontos consecutivos.",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig4, fig5

def main(argv=None):
    parser = argparse.ArgumentParser(description="Simulação do tópico 4: ruído em FM")
    parser.add_argument("--sem-janelas", action="store_true",
                        help="salva os cinco gráficos e dois CSVs sem abrir janelas")
    args = parser.parse_args(argv)
    import matplotlib
    if args.sem_janelas:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(int(fs * T)) / fs
    m = np.cos(2 * np.pi * fm * t)
    # Uma única sequência: AWGN (2), demodulação (2), varredura (460).
    # Não reutilizar ruído nem reiniciar o gerador entre as etapas.
    fig = validar_sem_ruido(plt, t, m)
    rng = np.random.default_rng(42)
    fig2 = demonstrar_awgn(plt, rng, t)
    fig3 = demonstrar_demodulacao(plt, rng, t, m)
    resultados_snr, t_varredura, m_varredura = varrer_snr(rng)
    analisar_limiar(resultados_snr, t_varredura, m_varredura)
    fig4, fig5 = graficos_snr(plt, resultados_snr)
    figuras = [fig, fig2, fig3, fig4, fig5]
    pasta = Path(__file__).resolve().parent / "resultados" / "topico4"
    pasta.mkdir(parents=True, exist_ok=True)
    salvar_resultados(figuras, resultados_snr, pasta)
    if args.sem_janelas:
        for figura in figuras:
            plt.close(figura)
    else:
        plt.show()
    return resultados_snr

if __name__ == "__main__":
    main()
