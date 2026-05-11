# `notebooks/` — Notebooks executáveis no Colab

Os notebooks aqui executam **treino** e **análise** dos modelos do TCC. O código-base permanece em `src/`; os notebooks são apenas a orquestração de uma sessão Colab + GPU.

## Pré-requisitos (uma vez por máquina)

1. **Conta Google** com Drive (aluno já tem: `thiagoguilherme123g@gmail.com`).
2. **Upload do dataset** `Data/` para o Drive em `MyDrive/TCC/Data/`. Estrutura:
   ```
   MyDrive/TCC/
   ├── Data/
   │   ├── Non Demented/
   │   ├── Very mild Dementia/
   │   ├── Mild Dementia/
   │   └── Moderate Dementia/
   └── runs/                    # criado automaticamente pelo notebook
   ```
   Volume: ~1,3 GB. Upload via Drive web ou Drive desktop (10-15 min).
3. **Repositório atualizado** em https://github.com/ThiagoCB1900/TCC . Sempre faça `git push` antes de rodar o notebook — o Colab clona da `main`.

## Por que symlinks?

Cada notebook cria symlinks dentro de `/content/TCC/` apontando para o Drive:
- `Data/` → `MyDrive/TCC/Data/` (leitura)
- `experiments/runs/` → `MyDrive/TCC/runs/` (escrita)

Assim **todos os checkpoints, history e métricas salvam direto no Drive durante o treino**. Se a sessão for revogada, nada se perde. Quando recomeçar a sessão, os outputs já estão lá para análise.

## Notebooks disponíveis

| Notebook | Objetivo | Tempo estimado (T4 grátis) |
|---|---|---|
| `01_resnet50_baseline_colab.ipynb` | Baseline ResNet-50 com pesos ImageNet (ADR-0008) + ablação sem peso (ADR-0007) | ~3-5h total (1 com peso + 1 sem peso, com early stopping) |
| *(futuros)* `02_vit_base_colab.ipynb` | ViT-Base/16 fine-tune | a planejar |
| *(futuros)* `03_swin_tiny_colab.ipynb` | Swin-T fine-tune | a planejar |
| *(futuros)* `04_interpretability_colab.ipynb` | Attention Rollout + Grad-CAM | a planejar |

## Como rodar (resumo)

1. Abra o notebook desejado no Colab (`File → Open notebook → GitHub → ThiagoCB1900/TCC`).
2. `Runtime → Change runtime type → GPU`.
3. Execute as células em ordem.
4. Na célula de "Montar Drive", autorize o Colab.
5. Acompanhe os logs no próprio notebook.

## Análise local dos resultados

Após o treino no Colab, baixe `MyDrive/TCC/runs/<run_id>/` para `experiments/runs/` localmente (ou puxe via Drive desktop). O `history.json` e `final_test_metrics.json` são pequenos e podem ser commitados; `checkpoint_best.pt` fica ignorado por `*.pt` no `.gitignore`.

## Reprodutibilidade

- Manifest e split estão versionados (`results/eda/manifest.csv`, `experiments/splits/split_v1.json`).
- Seeds fixas (`42`) em todas as decisões aleatórias (split, augmentation generator, torch seed).
- Configuração da run salva em `config.yaml` ao lado de cada `history.json`.
- Hash do commit (`git_commit` no config.yaml) registra exatamente qual versão do código gerou aquele resultado.

Para reproduzir uma run em outra máquina ou momento:
1. `git checkout <git_commit>` da config
2. Mesmo dataset (split JSON garante mesmos pacientes nos folds)
3. `python -m src.training.run` com os mesmos flags
4. Resultado bate (dentro de variação numérica de GPU não-determinística)
