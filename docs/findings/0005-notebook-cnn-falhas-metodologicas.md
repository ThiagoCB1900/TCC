# F-0005 — Notebook CNN comparativo (uraninjo dataset) — 5 falhas metodológicas

- **Data:** 2026-05-10
- **Status:** Confirmed
- **Categoria:** trabalho-relacionado

## O que descobrimos

O aluno incorporou ao projeto um notebook prévio em `CNN/Alzheimers disease dataset/Alzheimer.ipynb` para usar como comparativo no TCC. Após auditoria:

**Dataset usado:** `uraninjo/augmented-alzheimer-mri-dataset` (Kaggle), **diferente do nosso** (`ninadaithal/imagesoasis`):

| Propriedade | Notebook CNN | Nosso TCC |
|---|---|---|
| Dataset Kaggle | uraninjo/augmented-alzheimer-mri-dataset | ninadaithal/imagesoasis |
| Subject IDs preservados | **Não** (`26 (19).jpg`) | Sim (`OAS1_XXXX_*`) |
| Origem documentada | Disputada | OASIS-1 oficial (verificável) |
| Tamanho original | 6.400 imagens (Original) | 86.437 slices |
| Tamanho com augmentation | 33.984 (Augmented) | — (faremos só no dataloader) |

**Cinco falhas metodológicas identificadas no notebook:**

1. **Vazamento massivo train↔test.** Train = `AugmentedAlzheimerDataset/`, Test = `OriginalDataset/`. Como as augmentadas são derivadas das originais, **toda imagem do test set tem versões aumentadas no train** → modelo memoriza.
2. **Split aleatório por arquivo.** `flow_from_directory(... validation_split=0.2)` não agrupa por paciente. Sem subject IDs no dataset uraninjo, isso é literalmente impossível de fazer corretamente nesse dataset.
3. **Treino em 10% dos dados por época.** `steps_per_epoch = int(0.1*len(train_data))` + `EarlyStopping(monitor='accuracy', patience=3)` (em train accuracy, não validation) — overfit muito provável, validação não foi usada como gatilho.
4. **CNN básica sem transfer learning.** 2 conv layers (kernel **2×2**, incomum), 128 filtros cada, maxpool, flatten, dense(4). Sem dropout, sem batchnorm, sem regularização L2, sem pretrained weights.
5. **Métrica única e enviesada.** Reportou apenas `model_1.evaluate(train_data) → 0.8797 accuracy`. **Sobre o train set**, não test. E acurácia bruta com ModerateDemented em 64/6400 (1% do test) → modelo trivial que ignora Moderate já bate 99% nele.

## Evidência

- Notebook completo lido cell-by-cell em 2026-05-10.
- Cells relevantes:
  - cell-7: definição train_dir/test_dir
  - cell-10/11: ImageDataGenerator com `validation_split=0.2`, `rescale=1/255`
  - cell-17: arquitetura
  - cell-19: `steps_per_epoch=int(0.1*len(train_data))`
  - cell-21: `model_1.evaluate(train_data)` retornou 0.8797
- Contagem direta dos diretórios:
  - OriginalDataset: 896 / 64 / 3.200 / 2.240 (Mild/Moderate/Non/VeryMild)
  - AugmentedAlzheimerDataset: 8.960 / 6.464 / 9.600 / 8.960 (augmentation **101× para Moderate**)

## Implicação

Este notebook é **caso de estudo perfeito** para a seção "Trabalho relacionado" do TCC:

- **Justificativa empírica para o split por paciente** (ADR-0002): o notebook prévio mostra exatamente o tipo de erro que estamos evitando.
- **Experimento de ablação valioso**: rodar a CNN do notebook **com nossa metodologia rigorosa** (split por paciente OASIS, sem augmentation leak, métricas balanceadas). Esperamos queda significativa de 87% → ~60-75% — evidência empírica do problema.
- **Contraste de modelo**: ViT-Base + Swin (pretrained ImageNet) vs CNN básica (sem transfer) — mostra o ganho de transformers em domínio médico.
- **Contraste de interpretabilidade**: notebook não tem nenhuma. Nossa entrega de Attention Rollout + Grad-CAM é diferenciador clínico.

## Notas / armadilhas

- **Não tentar reproduzir os 87,97% do notebook**: a metodologia compromete a interpretação. Não é "estado da arte que precisamos superar" — é caso de estudo de o que **não** fazer.
- **Não confundir os dois datasets**: o nosso é OASIS-1 (`Data/`), o do notebook é uraninjo (`CNN/.../`). São fontes distintas. O split por paciente que vamos implementar serve só para o nosso.
- A pasta `CNN/.../Alzheimer's dataset/` tem ~40k imagens (~1.5GB). Ignorada no `.gitignore`; o `.ipynb` permanece versionado como fonte primária.
- Se o aluno quiser fazer o experimento de ablação ("CNN com metodologia rigorosa") como item opcional do TCC, ele entra na ordem de corte se houver atraso.
