# `notebooks/` — Notebooks executáveis

Os notebooks aqui orquestram **treino, timing e análise** dos modelos do TCC. O código-base permanece em `src/`; os notebooks são apenas a "shell" de cada ambiente (Kaggle, Colab, CPU local).

## Os 3 ambientes e quando usar cada

| Notebook | Ambiente | Propósito | Status |
|---|---|---|---|
| [`03_resnet50_baseline_kaggle.ipynb`](03_resnet50_baseline_kaggle.ipynb) | **Kaggle Notebooks** | **Treino primário** (ADR-0009) | **PRIMÁRIO** |
| [`02_resnet50_local_cpu_timing.ipynb`](02_resnet50_local_cpu_timing.ipynb) | CPU local + VSCode | Medir tempo real em CPU; sanity sem GPU | Suporte |
| [`01_resnet50_baseline_colab.ipynb`](01_resnet50_baseline_colab.ipynb) | Google Colab | Backup se Kaggle indisponível | Defasado — ver F-0011 |

### Por que Kaggle é primário?

Detalhado em [F-0011](../docs/findings/0011-drive-symlink-i-o-bottleneck-no-colab.md) e [F-0012](../docs/findings/0012-kaggle-notebooks-resolve-i-o-do-drive.md):

- Colab + Drive: cada `__getitem__` é uma chamada FUSE lenta. Treino completo não fechou 1 epoch antes de esgotar cota gratuita.
- Kaggle: o dataset `ninadaithal/imagesoasis` está em SSD local da sessão — sem gargalo. GPU T4/P100 gratuita com 30h/semana. Cota explícita.

Decisão formalizada em [ADR-0009](../docs/decisions/0009-kaggle-notebooks-como-ambiente-primario-de-treino.md), que substitui ADR-0003.

## Como usar — Kaggle (caminho primário)

1. **Conta Kaggle** (kaggle.com, gratuita).
2. Abrir o notebook diretamente do GitHub: no Kaggle, **+ Create → New Notebook → File → Import Notebook → GitHub** → cole `ThiagoCB1900/TCC` + caminho `notebooks/03_resnet50_baseline_kaggle.ipynb`.
   - Alternativa: baixe o `.ipynb` do GitHub e faça upload via **+ Create → New Notebook → File → Upload Notebook**.
3. No painel direito:
   - **+ Add Input** → procurar `ninadaithal/imagesoasis` → **Add**. Vai aparecer em `/kaggle/input/imagesoasis/`.
   - **Settings → Accelerator → GPU T4 x1** (ou P100).
   - **Settings → Internet → On** (necessário pra `git clone` do nosso repo).
4. Execute células em ordem. A célula 2 valida que o dataset Kaggle bate exatamente com a nossa EDA (67.222 + 13.725 + 5.002 + 488 = 86.437).
5. Ao final, baixe `runs_to_commit.zip` (célula 12) e commite localmente em `experiments/runs/`.

**Tempo esperado em T4 com SSD local:** ~1,5-2,5h para treino completo do baseline + ~1,5-2,5h para a ablação sem peso. Cabe nas 30h/semana gratuitas.

## Como usar — CPU local (timing/sanity)

Para medir o quão lento é treinar localmente sem GPU, ou validar pipeline em isolamento:

1. VSCode aberto em `c:\Users\thiag\TCC`, kernel `.venv` selecionado.
2. Abrir [`02_resnet50_local_cpu_timing.ipynb`](02_resnet50_local_cpu_timing.ipynb).
3. Célula 3 = **calibração** (~30s) → estima quanto tempo cada epoch vai levar. Pare e leia.
4. Se viável, célula 4 roda 1 epoch real (interromper a qualquer momento).

## Como usar — Colab (backup, defasado)

⚠️ [`01_resnet50_baseline_colab.ipynb`](01_resnet50_baseline_colab.ipynb) **ainda precisa do fix de F-0011** (copiar `Data.zip` para `/content/` antes do treino). Em sua forma atual, perde para Kaggle em facilidade. Use só se Kaggle estiver fora do ar.

Pré-requisito: upload de `Data/` (ou `Data.zip`) para `MyDrive/TCC/`. Pode ser via atalho se a pasta estiver em "Compartilhados comigo".

## Reprodutibilidade entre ambientes

Independente do notebook usado, o que garante reprodutibilidade é:

- `manifest.csv` versionado (`results/eda/`)
- `split_v1.json` versionado (`experiments/splits/`)
- `seed=42` fixa em todos os pontos aleatórios
- `git_commit` registrado no `config.yaml` de cada run

Cada run gera `experiments/runs/<run_id>/{config.yaml, history.json, final_test_metrics.json}` — esses **devem ser commitados** no repo após o treino (são pequenos). Checkpoints `.pt` ficam fora do git (ignorado por `*.pt`).
