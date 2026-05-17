# F-0012 — Kaggle Notebooks resolve nativamente o gargalo de I/O do F-0011

- **Data:** 2026-05-10
- **Status:** Confirmed (proposto pelo aluno, validado pela análise técnica abaixo)
- **Categoria:** metodologia

## O que descobrimos

A plataforma **Kaggle Notebooks** elimina nativamente os três problemas que tornaram a tentativa Colab inviável:

1. **Dataset já fica em SSD local** (`/kaggle/input/imagesoasis/`) — montado read-only direto no notebook com 1 clique, sem upload, sem Drive FUSE.
2. **GPU gratuita mais previsível** — 30h/semana de T4 ou P100, sessões de até 9h. Cota é por **tempo de GPU ativo**, não por "compute units" intangíveis.
3. **Persistência integrada** — `/kaggle/working/` mantém os outputs na versão salva do notebook; dá pra baixar ou versionar.

A causa-raiz do F-0011 (60k chamadas FUSE por epoch ao Drive) **não existe** no Kaggle: as imagens estão num SSD local, e o `OASISDataset` lê com latência de microssegundos por imagem, não centenas de milissegundos.

## Evidência

- Dataset `ninadaithal/imagesoasis` já está hospedado no Kaggle e foi a fonte original do nosso `Data/` (ver F-0001).
- Kaggle Notebooks suporta: GPU (Settings → Accelerator), internet (necessário para `git clone` do nosso repo), persistência de outputs em `/kaggle/working/` (20 GB).
- Comparação direta de I/O:

| Característica | Colab + Drive symlink | Kaggle Notebooks |
|---|---|---|
| Latência por imagem | ~50-300 ms (FUSE+rede) | ~0,1-1 ms (SSD local) |
| 60k imagens/epoch (só I/O) | ~50-300 min | **~10-60 s** |
| 1 epoch ResNet-50 T4 total | esperado: ~30-60 min (I/O dominava) | esperado: **~5-7 min** |
| Cota por hora | "unidades computacionais" opacas | 30h/semana, transparente |

## Implicação

**Kaggle Notebooks passa a ser o ambiente primário de treino** (formalizado em ADR-0009, que substitui ADR-0003). Colab e CPU local ficam como caminhos secundários (Colab para casos sem dataset Kaggle disponível; CPU local para smoke tests e desenvolvimento).

Notebook concreto criado em `notebooks/03_resnet50_baseline_kaggle.ipynb`. Estrutura adaptada:

- `data_root='.'` continua válido via symlink `/kaggle/working/TCC/Data → /kaggle/input/imagesoasis`.
- `experiments/runs/` mapeia para `/kaggle/working/runs/` (persistido na versão do notebook).
- `pip install` só dos deltas (timm, monai, tabulate, pypdf, torchmetrics) — PyTorch com CUDA já vem pré-instalado.

## Notas / armadilhas

- **A integridade do dataset Kaggle precisa ser validada na primeira célula do notebook.** Esperado: 67.222 `Non Demented` (sem as 16 duplicatas do upload Drive — F-0011). Se a contagem diferir, o `manifest.csv` versionado pode estar dessincronizado em relação ao dataset Kaggle atual e precisa regeração. Asserção automática na célula 4 do notebook 03.
- **Internet do notebook precisa estar ativada** (Settings → Internet → On) para `git clone`. Padrão Kaggle é OFF.
- **`/kaggle/working/` tem limite de 20 GB** — suficiente para nossos outputs (~poucas centenas de MB por run), mas atento se gerar muitos checkpoints intermediários.
- **Notebook salvo no Kaggle vira "version"** com snapshot de inputs+código+outputs. É um mecanismo natural de reprodutibilidade que substitui parcialmente o `git_commit` registrado no `config.yaml`.
- **Multi-GPU (T4 × 2)** está disponível para nossa carga mas é overkill (modelo cabe em 1 GPU); não vamos usar.
- **Limitação para futuro:** se mudarmos pra dataset OASIS-1 oficial completo (vs versão Kaggle), perde-se a vantagem de "dataset pronto". Mas estamos comprometidos com a versão Kaggle (F-0002), então não é problema.

## Comparativo final dos ambientes

| Ambiente | Função no TCC | Por quê |
|---|---|---|
| **Kaggle Notebooks** | **Primário para treino** | Dataset nativo + GPU gratuita estável (F-0012) |
| **CPU local + VSCode** | Dev, EDA, smoke test, escrita | Edição confortável; sem GPU mas pipeline roda |
| **Colab grátis** | Backup se Kaggle ficar indisponível | F-0011 corrigido (Data.zip → /content/) mas perde para Kaggle em facilidade |
| **Colab Pro** | Plano B se cota Kaggle esgotar antes da entrega | A100 fecha treino em <30 min; investimento se cronograma apertar |
