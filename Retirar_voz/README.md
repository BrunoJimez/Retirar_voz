# StemForge

Separação profissional de faixas musicais (vocal, bateria, baixo, guitarra,
piano, demais instrumentos) com **modelos de IA abertos, livres e gratuitos**,
rodando **100% localmente**. Nenhuma API paga, nenhuma chave, nenhum upload.

O programa inspeciona o hardware da máquina, analisa tecnicamente o áudio,
corrige o que estiver defeituoso, separa as fontes e exporta cada componente
em qualquer formato de áudio ou vídeo.

---

## 1. Modelos e licenças

| Componente | Projeto | Licença | Custo |
|---|---|---|---|
| Separação de fontes | Demucs v4 (Meta AI) | MIT | gratuito |
| Refino de vocal (opcional) | UVR / MDX-Net via `audio-separator` | MIT | gratuito |
| Redução de ruído | `noisereduce` | MIT | gratuito |
| Loudness EBU R128 | `pyloudnorm` | MIT | gratuito |
| Análise musical | `librosa` | ISC | gratuito |
| Codificação | FFmpeg | LGPL/GPL | gratuito |

Os pesos do Demucs são baixados uma única vez (~2 GB para `htdemucs_ft`) e
ficam em cache local. Depois disso o programa funciona offline.

---

## 2. Instalação

```bash
# 1. FFmpeg (obrigatório)
sudo apt install ffmpeg          # Debian/Ubuntu
brew install ffmpeg              # macOS
winget install Gyan.FFmpeg       # Windows

# 2. Ambiente Python
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip

# 3. PyTorch — escolha conforme sua placa
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121  # NVIDIA
pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm6.0 # AMD (Linux)
pip install torch torchaudio                                                      # CPU / Apple Silicon

# 4. StemForge
pip install -r requirements.txt
pip install -e .

# 5. Opcional: refino de vocal com modelos UVR
pip install "audio-separator[gpu]"   # ou [cpu]
```

Confira se tudo foi reconhecido:

```bash
stemforge --diagnostico
```

---

## 3. Uso

```bash
# Básico — 4 stems em wav, flac e mp3
stemforge musica.mp3

# 6 stems (inclui guitarra e piano) na qualidade máxima
stemforge musica.flac --6-stems --qualidade maxima

# Vídeo por stem, com espectrograma animado
stemforge musica.wav --video mp4 mkv --visual espectro --resolucao 1920x1080

# Só o vocal e o instrumental, em mp3
stemforge musica.mp3 --apenas vocals instrumental --formatos mp3

# Vídeo com a capa do álbum como fundo
stemforge musica.flac --video mp4 --capa capa.jpg

# Só o laudo técnico, sem separar
stemforge musica.mp3 --somente-analise
```

### Principais opções

| Opção | Efeito |
|---|---|
| `--qualidade rapido\|equilibrado\|maxima\|auto` | preset; `auto` decide pelo hardware |
| `--6-stems` | usa `htdemucs_6s` (guitarra e piano separados) |
| `--modelo NOME` | força `htdemucs`, `htdemucs_ft`, `htdemucs_6s`, `mdx_extra` |
| `--device cuda\|mps\|cpu` | força o dispositivo |
| `--refinar-vocal` | segunda passagem UVR/MDX-Net no vocal |
| `--formatos ...` | `wav wav16 wav32 flac aiff mp3 mp3v0 m4a aac opus ogg` |
| `--video ...` | `mp4 mkv mov webm` |
| `--visual ...` | `waveform espectro barras vetorscope preto` |
| `--apenas ...` | seleciona quais stems exportar |
| `--normalizar --lufs -14` | normaliza cada stem (desligado por padrão) |
| `--denoise 0.5` | força a redução de ruído na fonte |
| `--json` | relatório completo em JSON no stdout |

---

## 4. O que é gerado

```
saida/<nome-da-musica>/
├── vocal.{master.wav,wav,flac,mp3,mp4,mkv}
├── bateria.*
├── baixo.*
├── guitarra.*            (com --6-stems)
├── piano.*               (com --6-stems)
├── outros.*
├── instrumental.*        derivado: mixagem sem o vocal
├── karaoke.*             idem, nome alternativo
├── ritmica.*             derivado: bateria + baixo
├── harmonica.*           derivado: guitarra + piano + outros
├── vocal_centro.*        derivado: componente mid do vocal (voz principal)
├── vocal_lateral.*       derivado: componente side (dobras, backing, reverb)
└── relatorio.json        laudo técnico completo
```

O `*.master.wav` é sempre WAV 32-bit float — a versão sem perda e sem
truncamento, da qual todas as outras são derivadas. Nenhuma conversão é feita
em cascata.

---

## 5. Cadeia de processamento

```
hardware  →  análise  →  condicionamento da fonte  →  separação (IA)
          →  stems derivados  →  processamento por stem  →  exportação
```

**Análise** mede: pico, true peak (4× oversampling), RMS, LUFS integrado, LRA,
fator de crista, offset DC, clipping, piso de ruído por estatística de mínimos
espectrais, largura de banda efetiva (detecta fonte comprimida com perda),
correlação e largura estéreo, BPM e tonalidade (Krumhansl-Schmuckler).

**Condicionamento** só aplica o que o laudo pediu: remoção de DC, high-pass de
rumble, reconstrução de amostras clipadas por spline cúbica, redução de ruído
espectral, correção de polaridade.

**Pós-processamento por stem** repete a análise em cada faixa separada e aplica
correção mais conservadora (o denoise é limitado a 0.35 para não apagar cauda
de reverb), com limitador true-peak de lookahead antes da exportação.

Por padrão os stems **não são normalizados**: manter o balanço relativo
significa que somar todos eles reproduz a mixagem original. Use `--normalizar`
se quiser cada faixa em um alvo LUFS fixo.

---

## 6. Adaptação ao hardware

`--diagnostico` mostra o que foi detectado e por quê cada parâmetro foi
escolhido. As regras principais:

| Situação | Decisão automática |
|---|---|
| VRAM ≥ 16 GB | sem partição forçada, `htdemucs_ft`, shifts 4 |
| VRAM 8–16 GB | segmento 15 s, meia precisão se compute ≥ 7.0 |
| VRAM 6–8 GB | segmento 10 s |
| VRAM 4–6 GB | segmento 7 s |
| VRAM < 4 GB | segmento 5 s |
| Apple Silicon | backend MPS, segmento 10 s |
| Só CPU | jobs = min(núcleos físicos, (RAM−2)/2, 4) |
| RAM < 8 GB | segmento 6 s |

Se ocorrer *out of memory* durante a separação, o programa reduz o segmento
automaticamente até três vezes e, persistindo, migra para a CPU sem abortar.

---

## 7. Limites honestos

- A separação herda a qualidade da fonte. Um mp3 de 128 kbps tem corte em
  ~16 kHz; nenhum stem terá conteúdo acima disso. O laudo avisa quando detecta
  fonte comprimida.
- Nenhum modelo atual separa perfeitamente. Espere resíduo de "vazamento"
  entre faixas, sobretudo em masterizações muito comprimidas (o laudo também
  avisa quando o fator de crista está baixo).
- `vocal_centro` / `vocal_lateral` são uma decomposição mid/side, não uma
  separação por IA de voz principal e backing — funciona bem quando o lead
  está centralizado, o que é o caso na maioria das mixagens.
- `--qualidade maxima` com `htdemucs_ft` e shifts 4 custa cerca de 16× o tempo
  do preset rápido. Em CPU isso pode significar horas para uma faixa.

---

## 8. Uso como biblioteca

```python
from stemforge.pipeline import Config, executar

rel = executar(Config(
    entrada="musica.flac",
    saida="saida",
    seis_stems=True,
    qualidade="maxima",
    formatos_audio=["flac", "mp3"],
    formatos_video=["mp4"],
    visual="espectro",
))
print(rel["separacao"]["residuo_db"])
```

Ou módulo a módulo:

```python
from stemforge import hardware, analysis, enhance, separate, export

perfil = hardware.coletar_hardware()
plano  = hardware.planejar_execucao(perfil, "auto")
audio, sr, _ = analysis.carregar_audio("musica.wav")
laudo  = analysis.analisar("musica.wav", audio, sr)
audio, _ = enhance.processar(audio, sr, laudo.sugestoes, alvo_lufs=None)
res    = separate.separar(audio, sr, plano)
stems  = {**res.stems, **separate.derivar_stems(res.stems)}
```

---

Licença sugerida: MIT (compatível com todas as dependências acima).
