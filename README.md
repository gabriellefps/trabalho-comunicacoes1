# trabalho-comunicacoes1

Simulação do trabalho T3 de Comunicações I: modulação FM, espectro de Bessel,
largura de banda e demodulação por discriminador.

## Como executar

Dependências: Python 3, NumPy, SciPy e Matplotlib.

```bash
python3 -m pip install numpy scipy matplotlib
python3 fm-simulador.py
```

O simulador usa `audio.wav` por padrão. Para outro arquivo ou execução sem
abrir janelas:

```bash
python3 fm-simulador.py --audio audio.wav --sem-janelas
```

## Tópico 3 — Demodulador discriminador

O receptor está em `topico3.py` e é chamado pelo `fm-simulador.py` com os
mesmos sinais do tópico 1: tom de 1 kHz com β = 0,3; 1; 5; 10 e áudio
com β = 0,3 e 5. A validação desta etapa é feita **sem ruído**.

O código segue estas etapas:

1. **Limitador:** divide o sinal pelo envelope obtido com Hilbert, deixando
   a amplitude da portadora igual a 1.
2. **Derivada espectral:** multiplica a FFT por `j * 2 * pi * frequencia`.
   Isso transforma a variação de frequência em variação de envelope e evita
   a atenuação de ganho da diferença central (`np.gradient`).
3. **Detector de envelope:** calcula o módulo do sinal analítico da derivada.
4. **Remoção da portadora e escala:** usa
   `mensagem = (envelope / (2 * pi) - frequencia_portadora) / desvio_frequencia`.
   Aqui a subtração da portadora conhecida é adequada porque a derivada é
   espectral. A fórmula pressupõe mensagem normalizada e frequência
   instantânea positiva: `frequencia_portadora > desvio_frequencia`.
5. **Passa-baixas ideal:** zera as componentes da FFT acima da banda da
   mensagem, sem atenuar o tom na frequência de corte nem introduzir atraso.

A derivada, Hilbert e o filtro ideal operam sobre o sinal inteiro; este é um
receptor de simulação offline. A FFT pressupõe repetição periódica do trecho,
o que pode causar erros nas bordas do áudio. Por isso, o RMSE (raiz do erro
quadrático médio) e o erro relativo usam os 90% centrais. O erro relativo é
`100 * RMSE / RMS da mensagem original`, sem ajustar o ganho da recuperação.

No áudio, a integração por trapézios e o corte ideal do receptor também
introduzem pequenas diferenças: o pré-filtro Butterworth da mensagem não
elimina completamente as frequências acima de 3 kHz.

Com o `audio.wav` do repositório, o RMSE do tom ficou abaixo de `1,3e-12`.
O áudio apresentou RMSE de aproximadamente `2,76e-3` e erro relativo de
`2,01%` para ambos os valores de β. A diferença vem principalmente das
componentes acima de 3 kHz que o passa-baixas ideal remove: aplicar esse
mesmo corte diretamente à mensagem original já produz RMSE de `2,76e-3`.

O filtro passa-faixa de entrada não é necessário nesta validação sem ruído.
Para medir desempenho com AWGN no tópico 4, a filtragem do canal deve ocorrer
antes do limitador, e a banda usada na SNR deve ser definida explicitamente.

Os resultados novos ficam em `resultados/`:

- `06_demodulacao_tom.png`: mensagem original, recuperada e erro para cada β.
- `07_demodulacao_audio.png`: comparação no mesmo trecho de maior energia
  selecionado no tópico 1.
- `demodulacao.csv`: RMSE e erro relativo para tom e áudio.
- `audio_referencia.wav` e `audio_recuperado_beta_*.wav`: mesmos 90% centrais,
  em PCM de 16 bits, para comparação auditiva com o mesmo ganho. A limitação
  para o intervalo [-1, 1] ocorre apenas ao exportar os áudios recuperados;
  as métricas são calculadas antes dela.

Para verificar o discriminador:

```bash
python3 -m unittest test_topico3.py
```
