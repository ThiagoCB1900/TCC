# TCC — Vision Transformers para Classificação de Alzheimer

> **Para o Claude Code:** este arquivo é a fonte da verdade do projeto. É lido automaticamente em toda sessão. Como o aluno trabalha em **dois computadores**, mantenha-o atualizado a cada decisão relevante (modelo, hiperparâmetro, resultado, mudança de escopo). A memória local de Claude Code não sincroniza entre máquinas — este arquivo, sim.

---

## Identificação

- **Aluno:** Thiago Guilherme Aguiar (`thiagoguilherme123g@gmail.com`)
- **Curso:** Ciências da Computação · UECE
- **Tema:** Vision Transformers para Classificação de Alzheimer
- **Dataset:** OASIS-1 (já obtido pelo aluno)
- **Início do plano de 60 dias:** 2026-05-07
- **Entrega da cópia (orientador):** fim da semana 4 (~2026-06-03)
- **Defesa prevista:** fim da semana 8 (~2026-07-06)

## Estado atual

- **Fase:** Semana 1 · Fundação · ambiente configurado, EDA em execução
- **Estrutura de pastas criada:** sim (ver "Layout do repositório" abaixo)
- **Ambiente Python:** `.venv/` no projeto, Python 3.11, PyTorch 2.4.1 **CPU** + timm 1.0.11 + MONAI 1.3.2 (ver `requirements.txt`).
- **Hardware:** aluno tem RX6600 (AMD). Em Windows o suporte de PyTorch para AMD é limitado (DirectML experimental; ROCm é só Linux e nem suporta a 6600). **Decisão (ADR-0009): EDA e dev em CPU local; treino do baseline e modelos no Kaggle Notebooks (T4/P100 gratuita com dataset OASIS-Kaggle já hospedado em SSD local). Colab é backup após fix do F-0011.**
- **EDA executada (2026-05-10):** outputs em `results/eda/` (manifest.csv com 86.437 linhas, summary.json, eda_report.md, 7 figuras).
- **Split por paciente executado (2026-05-10):** `src/data/splits.py` produz `experiments/splits/split_v1.json` reprodutível (seed=42, 70/15/15, estratificado por class_3). 242 train / 52 val / 53 test sujeitos; proporções dentro de ~1% da distribuição global em todas as classes. Validações automáticas (sets disjuntos, cobertura de classes) passaram.
- **Dataset PyTorch implementado (2026-05-10):** `src/data/dataset.py` com `OASISDataset` (lê manifesto + split JSON, filtra por fold, label encoding por severidade clínica) + `build_dataloaders`. Pipeline: resize 224×224 squash + ImageNet norm + augmentations leves só no train (flip horizontal, rotação ±5°, jitter brilho/contraste). Validado em `src/data/inspect_dataset.py` com inspeção visual em `results/eda/figures/dataset_inspection.png` e `dataset_augmentation_check.png` — checkpoint anti-erro motivado por F-0003.
- **Literatura auditada (2026-05-10):** varredura sistemática nos 14 PDFs em `docs/Papers/Trabalhos Relacionados/` via `src/data/survey_imbalance_handling.py`. Resultados em F-0008 (RanCom-ViT no mesmo dataset, 99,54% inflado por split inválido) e F-0009 (síntese de como ViT+Alzheimer trata desbalanceamento). Decisão fundamentada em ADR-0007: weighted CE 'balanced' + ablação sem peso.
- **Baseline ResNet-50 implementado + smoke test validado (2026-05-10):** `src/models/resnet.py` (timm) + `src/evaluation/metrics.py` (macro-F1, balanced acc, AUC, McNemar) + `src/training/{config,loop,run}.py`. Decisões em ADR-0008 (PyTorch puro + torchmetrics, sem Lightning). Smoke test rodou em CPU local em 2 min, validado em `experiments/runs/20260510_205637_resnet50_smoke_smoke/`. **Bug detectado e corrigido durante validação** (F-0010): `max_batches` em eval + ordem agrupada do manifesto = matriz de confusão inválida; corrigido com `shuffle_eval` em smoke. Pipeline pronto para treino real no Colab.
- **Notebook Colab criado (2026-05-10):** `notebooks/01_resnet50_baseline_colab.ipynb` executa todo o protocolo (Drive → clone → symlinks → smoke → treino com peso → treino sem peso → análise). Protocolo documentado em `notebooks/README.md`. Outputs persistidos em `MyDrive/TCC/runs/` via symlink, sobrevivem a queda de sessão.
- **Tentativa de treino no Colab gratuito falhou (2026-05-10):** smoke test OK no T4, mas treino completo não completou nem 1 epoch antes de esgotar cota de unidades computacionais. Causa diagnosticada em F-0011: `Data/` lido via symlink direto do Drive ⇒ 60k chamadas FUSE individuais/epoch ⇒ I/O domina ⇒ GPU fica idle queimando cota. **Notebook 01 precisa ser corrigido** (copiar `Data.zip` para `/content/Data/` no início da sessão).
- **Notebook local para timing criado (2026-05-10):** `notebooks/02_resnet50_local_cpu_timing.ipynb` mede tempo real de 1 epoch em CPU local enquanto cota Colab restaura. Inclui calibração de 3 batches que estima total antes de comprometer horas.
- **Mudança de ambiente primário (2026-05-10, ADR-0009):** aluno sugeriu Kaggle Notebooks. Análise (F-0012) confirma — dataset `ninadaithal/imagesoasis` já hospedado lá em SSD local da sessão (sem FUSE), GPU T4/P100 com 30h/semana, cota explícita. `notebooks/03_resnet50_baseline_kaggle.ipynb` é o novo primário; notebook 01 (Colab) ganha banner apontando pro 03.
- **Baseline V1 executado no Kaggle (2026-05-17):** 2 runs reais — weighted CE (bal_acc=0,623; macro_F1=0,554; AUC=0,804) e ablação sem peso (bal_acc=0,497). **Weighted CE empiricamente validado** (ADR-0007): F1 da classe minoritária subiu de 0,17 para 0,40 (+135%). Documentado em F-0013. **Mas overfit catastrófico observado**: train_loss colapsa 30× em 4 epochs; best epoch=1 → LR alto demais para o tamanho efetivo do dataset.
- **Baseline V2 implementado (2026-05-17, ADR-0010):** hiperparâmetros revisitados para combater overfit. LR diferenciado (backbone 1e-5, head 1e-4), `drop_rate=0.3`, `weight_decay=0.1`, augmentation forte (rot ±15°, jitter 0.2, translation 5%), `patience=5`, `warmup=2`. Compatibilidade retroativa preservada via flags (`--uniform-lr --augment light --drop-rate 0.0 ...` reproduz V1). Smoke V2 validou pipeline (param groups corretos: backbone 23.5M / head 6.1k; LR diferenciado aplicado). Notebook 03 atualizado.
- **Baseline V2 executado no Kaggle (2026-05-22, F-0017):** 2 runs — V2 weighted (bal_acc=0,587; macro_F1=0,513; AUC=0,804; best epoch=11) e V2 noweight (bal_acc=0,546; AUC=0,821; best epoch=9). **Overfit CONTROLADO** (train_loss cai gradual em 16 ep vs colapso em 4 ep do V1; val_loss estável; best epoch 1→11). **Mas métricas não subiram:** teto do ResNet-50 baseline ≈ 0,59-0,62 balanced_acc. O 0,623 do V1 era instável (epoch 1, pré-colapso); V2 weighted (0,587) é o **baseline oficial** (treino estável, número confiável). Gargalo deixou de ser overfit → agora é representação (fatia 2D única + ResNet). Alavanca para subir: dados (2.5D) ou arquitetura (ViT/Swin), não mais regularização.
- **ViT-Base/16 e Swin-T treinados (2026-05-23, F-0018):** transformers VOLTARAM a overfittar rápido (best epoch 2; train colapsa; val explode) com hiperparâmetros V2. **Swin-T é o melhor modelo** (test bal_acc 0,594 > ResNet 0,587 > ViT 0,521); Swin > ViT com significância (McNemar p=0,017). AUC transformers (0,82) > ResNet (0,80); val pico Swin 0,668 (melhor do projeto) → ganho preso atrás do overfit. ViT-Base subaproveitado (overfit severo, 86M params).
- **Próximo passo (decisão pendente):** (1) **regularizar transformers** ("V3-transformer": lr_backbone 5e-6, drop_path 0,2-0,3, weight_decay 0,2-0,3, warmup 3-5 — tudo via CLI, sem mudar código) para destravar o overfit; depois (2) **2.5D** (ADR-0011) no melhor modelo. Re-rodar ResNet V2 com persistência de predições para McNemar das 3 arquiteturas.

## Resultados consolidados (test set, split_v1, 3 classes, weighted CE)

| Modelo | accuracy | balanced_acc | macro_f1 | auc_macro | best_ep | obs |
|---|---:|---:|---:|---:|---:|---|
| ResNet-50 V1 weighted | 0,699 | 0,623* | 0,554 | 0,804 | 1 | *instável (epoch 1, pré-colapso) |
| **ResNet-50 V2 weighted** | 0,681 | 0,587 | 0,513 | 0,804 | 11 | baseline oficial (treino estável) |
| ViT-Base/16 weighted | 0,730 | 0,521 | 0,483 | 0,822 | 2 | overfit severo (F-0018) |
| **Swin-T weighted** | 0,723 | **0,594** | **0,528** | **0,823** | 2 | **melhor atual**; overfit (val pico 0,668) |

Comparação honesta: RanCom-ViT reporta 99,54% acc no mesmo dataset com split por slice (F-0008) — nosso ~0,59-0,72 acc com split por paciente é a métrica honesta. McNemar ViT vs Swin: p=0,017 (Swin significativamente melhor). Teto persistente ~0,59-0,67 → gargalo de dados (fatia 2D única) + regularização insuficiente dos transformers.

## Base de dados — versão Kaggle pré-processada (mudança importante)

A base original do plano era OASIS-1 bruto (NIfTI 3D + scripts FSL/BET). O aluno baixou a **versão Kaggle pré-processada** já em fatias 2D JPG, salvas em `Data/`. Isso muda o pipeline:

- **Pré-processamento já feito** pelo dataset Kaggle (skull stripping + slicing axial). Pula-se a etapa FSL BET.
- **Formato:** JPG 2D, fatias axiais centrais (índices 100–160 do volume).
- **Nomenclatura:** `OAS1_XXXX_MR1_mpr-N_NNN.jpg`
  - `OAS1` = OASIS-1
  - `XXXX` = subject ID (chave do split por paciente)
  - `MR1` = sessão 1 (única em OASIS-1)
  - `mpr-N` = N-ésima aquisição MPRAGE T1 dentro da sessão (N=1..4 = replicatas para reduzir ruído — **não são pacientes diferentes**)
  - `NNN` = índice da fatia axial (100–160 = 61 fatias por aquisição)

**Contagem encontrada (a confirmar com EDA executada):**

| Pasta (rótulo Kaggle) | Pacientes | Slices | % |
|---|---:|---:|---:|
| Non Demented | 266 | 67.222 | 77,8% |
| Very mild Dementia | 58 | 13.725 | 15,9% |
| Mild Dementia | 21 | 5.002 | 5,8% |
| Moderate Dementia | **2** | 488 | 0,6% |
| **Total** | **347** | **86.437** | 100% |

**Implicações críticas:**
1. **Moderate Dementia tem só 2 pacientes** — split por paciente em 4 classes é inviável (test set teria 0–1 paciente em Moderate).
2. **Desbalanceamento extremo** (138× entre maior e menor classe): acurácia bruta vai mentir; usar **macro-F1, balanced accuracy, AUC** como métricas primárias.
3. Cada paciente tem ~4 aquisições mpr × ~61 fatias = **~244 slices/paciente**. Confirma que split por slice criaria vazamento gigante — split **por subject ID** é mandatório.

## Decisões obrigatórias (revisar até o fim da semana 1)

| Decisão | Valor atual | Status | ADR |
|---|---|---|---|
| Versão do OASIS | OASIS-1 (Kaggle pré-processado, 2D JPG) | **fixado** | F-0001/F-0002 |
| Esquema de classes | **3 classes:** Non Demented / Very mild / Mild+Moderate (fundidos) | fixado · confirmar com orientador | [ADR-0001](docs/decisions/0001-esquema-de-classes-3-classes.md) |
| Split | Estratificado por subject ID, 70/15/15, seed fixa | **fixado** | [ADR-0002](docs/decisions/0002-split-estratificado-por-paciente.md) |
| Hardware de treino | **Kaggle Notebooks** (T4 grátis, dataset nativo); Colab/local como backups | **fixado** | [ADR-0009](docs/decisions/0009-kaggle-notebooks-como-ambiente-primario-de-treino.md) (substitui ADR-0003) |
| Pré-processamento | Resize squash 224×224, RGB sintético mantido | **fixado** | [ADR-0004](docs/decisions/0004-preprocessamento-resize-224.md) |
| Métricas primárias | macro-F1, balanced accuracy, AUC, McNemar | **fixado** | [ADR-0005](docs/decisions/0005-metricas-primarias.md) |
| Augmentations + label encoding | Flip H + rot ±5° + jitter leve só no train; classes 0=non, 1=very_mild, 2=mild_or_moderate | **fixado** | [ADR-0006](docs/decisions/0006-dataset-augmentations-e-label-encoding.md) |
| Tratamento de desbalanceamento | Weighted CE 'balanced' (sklearn) com ablação sem peso | **fixado** | [ADR-0007](docs/decisions/0007-tratamento-de-desbalanceamento.md) |
| Pipeline de dados (consolidado) | Split paciente → aug train-only → resize 224 squash → ImageNet norm; balanceamento e augmentation NUNCA em val/test | **fixado** | [ADR-0011](docs/decisions/0011-pipeline-de-dados-definitivo.md) |
| Modelos comparados | ViT-Base/16 (`vit_base_patch16_224`) + Swin-T (`swin_tiny_patch4_window7_224`) | **fixado** | [ADR-0012](docs/decisions/0012-modelos-transformer-vit-base-swin-tiny.md) |
| Baseline obrigatório | ResNet-50 com pesos ImageNet | **fechado** (F-0017) | — |

## Regras críticas — NÃO violar

1. **Split por paciente (subject ID), nunca por slice.** Dividir slices aleatoriamente faz com que o mesmo paciente apareça em treino e teste, inflando a acurácia artificialmente e invalidando o trabalho.
2. **End-to-end primeiro, qualidade depois.** Pipeline completo (dados → modelo → métricas) deve rodar até o fim da semana 2, mesmo com performance ruim.
3. **Nunca cortar:** baseline ResNet-50, split por paciente, interpretabilidade básica (Attention Rollout). São os três pilares que diferenciam o trabalho.
4. **Validar manualmente** todo código de pré-processamento, split e métricas — o aluno precisa defender cada decisão na banca.

## Ordem de corte se houver atraso

1. Experimento "CNN do notebook comparativo com nossa metodologia" (era opcional desde sempre — apenas se sobrar tempo após validar pipeline ViT/Swin)
2. Swin Transformer (mantém ViT-Base + ResNet)
3. Experimento de transfer learning sem pré-treino (mantém só fine-tuning ImageNet)
4. GradCAM no Swin (mantém só Attention Rollout no ViT)

## Stack técnica

- **Linguagem:** Python 3.11
- **Ambiente:** `.venv/` na raiz do projeto. Recriar em qualquer máquina com `python -m venv .venv && pip install -r requirements.txt`.
- **Frameworks:** PyTorch 2.4.1 (CPU local; CUDA no Colab), [timm](https://github.com/huggingface/pytorch-image-models) 1.0.11 para modelos, [MONAI](https://monai.io/) 1.3.2 para utilidades de imagem médica.
- **Pré-processamento:** já feito pela versão Kaggle (skull-stripped + slicing axial). FSL/BET **não será necessário** salvo se o orientador pedir reprocessamento do bruto.
- **Treino:** Colab (T4/A100). Local serve para EDA, dataloaders, smoke tests e interpretabilidade pós-treino.
- **Tracking de experimentos:** planilha simples (data, modelo, hiperparâmetros, métricas, observações). Nada mais sofisticado é necessário no escopo do TCC.

## Papers em `docs/Papers/Trabalhos Relacionados/`

Já baixados pelo aluno e disponíveis localmente como contexto:

- **ViT** — Dosovitskiy et al., 2020 (`AN IMAGE IS WORTH 16X16 WORDS.pdf`)
- **Swin** — Liu et al., 2021 (`Liu_Swin_Transformer_*.pdf`)
- **ResNet** — He et al., 2016 (`He_Deep_Residual_Learning_*.pdf`)
- **Transformer base** — Vaswani et al., 2017 (`NIPS-2017-attention-is-all-you-need-Paper.pdf`)
- **Grad-CAM** — Selvaraju et al., 2017
- **Attention Rollout / Flow** — Abnar & Zuidema (`attention_flow_in_transformers.pdf`)
- **Chefer Interpretability** — CVPR 2021 (alternativa moderna a Rollout puro)
- **Survey** — `Transformers_in_medical.pdf` (provável Shamshad et al.)
- **Aplicações Alzheimer + Transformer:** `An_efficient_vision_transformer_for_Alzheimers...`, `Joint_transformer_architecture_in_brain_3D_MRI_classification`, `Leveraging_Swin_Transformer_..._diffusion_MRI`, `VGG-TSwinformer`, `Vision_transformer_equipped_CNN`, `detection_of_Alzheimer_Disease_in_Neuroimages`

## Layout do repositório

```
TCC/
├── Data/                    # OASIS Kaggle (JPGs por classe) — NÃO commitar (.gitignore)
├── CNN/                     # notebook comparativo (.ipynb commitado, imagens .gitignored)
├── notebooks/               # exploração interativa
├── src/
│   ├── data/                # eda.py, inspect_image_layout.py, splits, dataloaders
│   ├── models/              # ViT, Swin, ResNet
│   ├── training/            # loops de treino
│   ├── interpretability/    # attention rollout, gradcam
│   └── evaluation/          # métricas, statistical tests (McNemar)
├── experiments/
│   └── splits/              # splits versionados (split_v1.json, ...) — commitar; configs e logs de runs entram aqui também
├── results/
│   └── eda/                 # manifest.csv, summary.json, eda_report.md, figures/
├── docs/
│   ├── decisions/           # ADRs (Architecture Decision Records)
│   ├── findings/            # Descobertas factuais auditadas
│   └── Papers/              # PDFs (em "Trabalhos Relacionados/")
├── .venv/                   # ambiente Python local (não commitado)
├── requirements.txt         # depend. pinadas
├── CLAUDE.md                # fonte da verdade do projeto (mapa, não território)
└── README.md
```

## Cronograma resumido (60 dias)

- **Sem 1 (dias 1-7):** fundação — leituras (ViT, Swin, survey Shamshad), ambiente, repositório, decisões fechadas com orientador.
- **Sem 2 (dias 8-14):** dados + baseline — download OASIS, EDA, pré-processamento, split por paciente, ResNet-50 baseline rodando.
- **Sem 3 (dias 15-21):** ViTs — fine-tuning ViT-Base e Swin-T, comparação com baseline, McNemar.
- **Sem 4 (dias 22-30):** interpretabilidade + entrega da cópia — Attention Rollout, GradCAM, análise qualitativa, entrega ao orientador no dia 30.
- **Sem 5-6 (dias 31-45):** refinamento — incorporar feedback, experimentos extras, escrita de resultados/discussão.
- **Sem 7-8 (dias 46-60):** fechamento — conclusão, revisões, slides (começar até dia 50), ensaios da defesa, defesa.

## Gotchas encontrados (atualizar à medida que aparecem novos)

1. **Imagens 2D são alongadas (496×248) mas contêm UM ÚNICO slice axial.** Aspect ratio ~2:1 vem da conversão NIfTI→JPG do uploader Kaggle (provável resampling com voxels não-isotrópicos). Não há concatenação de vistas. Slices 100-110 mostram o cérebro próximo à base, com globos oculares visíveis — **não confundir com vista sagital**. Decisão fixada: **resize direto para 224×224 (ViT-Base) ou 384×384 (Swin), squash para quadrado** (padrão de literatura em ViT médico).
2. **RGB sintético:** modo é RGB mas R==G==B em todas as imagens (grayscale duplicado pela conversão JPG). Para timm/ViT pré-treinados em ImageNet vamos manter os 3 canais (compatibilidade), sabendo que há 1 canal real de informação.
3. **MR2 existe (19 sujeitos non_demented):** é o sub-conjunto de reliability do OASIS-1 original (Marcus et al. 2007 — controles escaneados duas vezes para avaliar reprodutibilidade). Não é OASIS-2/longitudinal. Sem leak de classe entre MR1 e MR2. Para o split por paciente, agrupar por subject ID já é suficiente.
4. **mpr vai até 6 em 1 sujeito (`OAS1_0254`):** outlier benigno. OASIS-1 oficialmente tem 3-4 aquisições T1, mas esse sujeito teve 6. Sem ação necessária.
5. **`n_acquisitions` no manifesto deve agrupar por (subject, session, mpr)** — não só (subject, mpr) — senão sub-conta os MR2 dos 19 sujeitos. Bug pego e corrigido na primeira execução da EDA.
6. **`.gitignore` `Data/` precisa de barra inicial (`/Data/`):** sem âncora, a regra casa com `src/data/` no Windows (case-insensitive) e ignora silenciosamente os scripts de EDA. Detalhes em F-0006.
7. **Discrepância de pacientes Kaggle vs realidade:** o texto do Kaggle anuncia "461 patients" mas o que está em `Data/` são **347 pacientes** (IDs OAS1_0001 a OAS1_0382, com 35 buracos internos + IDs ≥383 ausentes). OASIS-1 oficial tem 416 sujeitos (Marcus 2007), então estamos com ~83% do dataset oficial. Provável causa: uploader excluiu pacientes sem CDR válido. **Não é download incompleto** — é o que o Kaggle disponibilizou. Documentar essa limitação no TCC.
8. **Val/test sem shuffle agrupam classes nos primeiros batches.** Manifesto está ordenado por classe; sem shuffle (correto, para determinismo de avaliação), `next(iter(val_loader))` pode mostrar só uma classe. Não é bug — métricas agregadas no fold permanecem corretas. Detalhes em F-0007.

## Sistema de anotações — ADRs e Findings

Toda **decisão metodológica** (split, classes, métricas, modelo, preprocessing, hardware...) vai como ADR em `docs/decisions/NNNN-titulo.md`. Toda **descoberta factual auditada** (estrutura do dataset, comportamento de imagem, achado em trabalho relacionado, anomalia) vai como Finding em `docs/findings/NNNN-titulo.md`. Os índices ficam nos READMEs respectivos.

**FAQ de defesa:** `docs/faq-defesa-banca.md` reúne respostas curtas e defensáveis para as perguntas conceituais mais prováveis na banca (balanceamento, augmentation/leakage, métricas, split por paciente, 3 classes, OASIS vs ADNI, etc.), cada uma ancorada no ADR/Finding correspondente. Atualizar quando surgir nova decisão relevante.

**Regra:** este `CLAUDE.md` é o **mapa**, não o **território**. Detalhes profundos vão para ADRs/Findings; o CLAUDE.md mantém apenas o resumo executivo e ponteiros. Quando houver conflito ou questão sobre "por que decidimos X", o ADR/Finding correspondente é a fonte canônica.

**Para o Claude Code (instruções de manutenção):**

- Antes de propor uma decisão metodológica nova, **leia** `docs/decisions/README.md` e os ADRs relevantes — pode haver decisão prévia que cobre o caso.
- Antes de fazer afirmação factual sobre o dataset, **leia** o Finding correspondente (ou crie um novo se não existir).
- Quando uma decisão antiga for revisitada, **não apague o ADR original** — crie um novo com `Status: Supersedes ADR-XXXX` e marque o antigo como `Superseded`.
- Adicione 1 linha no índice do README correspondente sempre que criar ADR ou Finding novo.
- Se um Finding sustentar uma decisão, **referencie-o explicitamente no ADR**.

## Trabalho relacionado / comparativo

Dois trabalhos serão usados como comparativos diretos no TCC:

1. **Notebook CNN do aluno** (`CNN/Alzheimers disease dataset/Alzheimer.ipynb`): dataset `uraninjo/augmented-alzheimer-mri-dataset` (diferente do nosso, sem subject IDs); **5 falhas metodológicas** documentadas em **[F-0005](docs/findings/0005-notebook-cnn-falhas-metodologicas.md)**.
2. **RanCom-ViT** (Lu, Zhang & Yao, *Biomedical Signal Processing and Control*, 2025): paper publicado que usa **EXATAMENTE o mesmo dataset Kaggle que o nosso** e reporta 99,54% acurácia. Auditado em **[F-0008](docs/findings/0008-rancom-vit-mesmo-dataset-metodologia-comprometida.md)**: split aleatório por slice (não por paciente), descrição errada do plano de slicing (chama de "sagital" o que é axial), métricas sem balanced accuracy.

Ambos os trabalhos são casos de estudo de prática não-rigorosa que justificam empiricamente nossas escolhas metodológicas. **Não almejar superar os 99,54% do RanCom-ViT** — é artefato metodológico. Nossa baseline será comparada principalmente com **ela mesma** (ablações: com vs sem weighted loss, com vs sem augmentation).

**Auditoria-mestre de split por paciente (2026-05-17, F-0014):** leitura integral dos 14 PDFs + auditoria do dataset uraninjo (scripts `src/data/extract_paper_texts.py` e `src/data/audit_cnn_dataset.py`). Quadro: **(a) trabalhos rigorosos fazem split por paciente** — Leveraging Swin, VGG-TSwinformer e Vision-ViT-CNN afirmam isso textualmente, e o Leveraging Swin **critica explicitamente** random splits ("evaluation design can inflate headline metrics"); **(b) RanCom-ViT** (mesmo dataset que nós) faz split por slice → 99,54% inflado; **(c) Joint Transformer** faz split por volume mas não controla leakage longitudinal (ADNI); **(d) dataset uraninjo (CNN do aluno) torna split por paciente IMPOSSÍVEL** — sem subject ID, EXIF vazio, prefixo numérico repetido entre classes. A tese do TCC é nuançada: não "todos erram", mas "os rigorosos fazem split por paciente; quem usa o nosso dataset ou datasets sem ID, não". Nosso diferencial: split por paciente rigoroso no OASIS-Kaggle.

**Mapa de posicionamento e gap (2026-05-17, F-0015):** métricas auditadas — RanCom-ViT 99,54% acc (OASIS, slice); Joint Transformer 99,05% (ADNI 3D, volume); Leveraging Swin bal_acc 95,2% (ADNI dMRI, sujeito); Vision-ViT-CNN 92,14% (ADNI 3D, sujeito); VGG-TSwinformer 77,2% (ADNI, sujeito, pMCI/sMCI). **OASIS só aparece em 1 trabalho de classificação (RanCom-ViT, split por slice); todos os 4 que fazem split por paciente usam ADNI.** Logo a interseção **ViT + OASIS + split por paciente está vazia** entre os trabalhos auditados — é o gap/contribuição do TCC. **Oponente direto = RanCom-ViT** (contraste metodológico no mesmo dataset, não disputa de número). Comparar números entre datasets/modalidades/splits diferentes é inválido — comparar contra nós mesmos (ablações + McNemar). Limite: gap confirmado entre os 14 papers do projeto; busca sistemática (Scholar/PubMed) recomendada antes da escrita final.

**Pipeline de dados fechado (2026-05-17, F-0016 + ADR-0011):** auditoria das práticas dos 14 papers + best-practices externas com fontes. **Regra de ouro confirmada (sua intuição estava certa): balanceamento e augmentation são SÓ no treino; val/test NUNCA são tocados** (balancear test → leakage, viés até 0,34 AUC por Nature Sci Reports; test reflete distribuição clínica real, desbalanceamento tratado na métrica). Vision-ViT-CNN (trabalho mais comparável) usa weighted loss = nosso ADR-0007. Código já conforme (aug só train, pesos sobre train fold, métricas sem peso). Pipeline fixo: split paciente → aug train-only → resize 224 squash → ImageNet norm + weighted CE. Ablações opcionais reservadas (só depois do baseline V2, uma por vez via McNemar): (1) **2.5D** slices vizinhos como canais RGB (maior potencial); (2) enhancement CLAHE; (3) augmentation de intensidade. Rejeitados: oversampling de slices, Nyúl/WhiteStripe (JPG), z-score (quebra ImageNet), TTA, qualquer aug/balance em val/test.

A pasta `CNN/.../Alzheimer's dataset/` contém ~40k imagens (~1.5GB). **Imagens ignoradas no `.gitignore`**, mas o `.ipynb` permanece versionado como fonte primária. Não modificar conteúdo da pasta — é fonte primária.

## Princípios operacionais

1. **End-to-end primeiro, qualidade depois.**
2. **Comece a escrever cedo** — introdução já na semana 2.
3. **Versione tudo desde o dia 1** (Git para código; planilha para experimentos).
4. **Reuniões semanais com orientador são inegociáveis** — enviar resumo escrito antes.
5. **Reserve buffer mental** — algo vai dar errado (GPU, OASIS, convergência); folga deliberada existe, não a queime no começo.

## Como manter este arquivo (instruções para o Claude Code)

- Sempre que houver decisão tomada, modelo escolhido, hiperparâmetro fixado, resultado relevante, ou mudança de escopo, **atualize a seção correspondente** deste arquivo **e crie ADR/Finding correspondente** em `docs/decisions/` ou `docs/findings/`.
- Atualize "Estado atual" com a fase atual e o próximo passo imediato sempre que avançar.
- Mantenha a tabela de "Decisões obrigatórias" refletindo o que já foi fechado vs. ainda em aberto. Cada linha "fixada" deve ter um ADR correspondente.
- Adicione uma seção "Resultados consolidados" assim que houver primeiras métricas (ResNet baseline, ViT, Swin) — formato tabela.
- "Gotchas" continua sendo um resumo executivo dos achados; detalhe e evidência vão para `docs/findings/`.
- Não duplicar o checklist completo de 60 dias — ele está no PDF original do plano. Aqui mantemos só o resumo e o estado.

## Referências de papers (a consultar em src/docs/)

- Dosovitskiy et al., 2020 — *An Image is Worth 16×16 Words* (ViT)
- Liu et al., 2021 — *Swin Transformer*
- Shamshad et al. — survey de transformers em imagem médica
- He et al., 2015 — ResNet (baseline)
- Selvaraju et al., 2017 — Grad-CAM
- Abnar & Zuidema, 2020 — Attention Rollout
