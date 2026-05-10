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
- **Hardware:** aluno tem RX6600 (AMD). Em Windows o suporte de PyTorch para AMD é limitado (DirectML experimental; ROCm é só Linux e nem suporta a 6600). **Decisão: EDA e dev em CPU local; treino do baseline e modelos em Colab (T4/A100).**
- **EDA executada (2026-05-10):** outputs em `results/eda/` (manifest.csv com 86.437 linhas, summary.json, eda_report.md, 7 figuras).
- **Split por paciente executado (2026-05-10):** `src/data/splits.py` produz `experiments/splits/split_v1.json` reprodutível (seed=42, 70/15/15, estratificado por class_3). 242 train / 52 val / 53 test sujeitos; proporções dentro de ~1% da distribuição global em todas as classes. Validações automáticas (sets disjuntos, cobertura de classes) passaram.
- **Próximo passo imediato:** implementar `src/data/dataset.py` (Dataset + DataLoader PyTorch consumindo `split_v1.json`, resize 224×224 squash, normalização ImageNet, augmentations leves no train), depois baseline ResNet-50.

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
| Hardware de treino | Colab (T4 grátis ou Pro) — local apenas para dev | **fixado** | [ADR-0003](docs/decisions/0003-hardware-colab-para-treino.md) |
| Pré-processamento | Resize squash 224×224, RGB sintético mantido | **fixado** | [ADR-0004](docs/decisions/0004-preprocessamento-resize-224.md) |
| Métricas primárias | macro-F1, balanced accuracy, AUC, McNemar | **fixado** | [ADR-0005](docs/decisions/0005-metricas-primarias.md) |
| Modelos comparados | ViT-Base/16 + Swin-T | a confirmar | — |
| Baseline obrigatório | ResNet-50 com pesos ImageNet | fixo do plano | — |

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

## Sistema de anotações — ADRs e Findings

Toda **decisão metodológica** (split, classes, métricas, modelo, preprocessing, hardware...) vai como ADR em `docs/decisions/NNNN-titulo.md`. Toda **descoberta factual auditada** (estrutura do dataset, comportamento de imagem, achado em trabalho relacionado, anomalia) vai como Finding em `docs/findings/NNNN-titulo.md`. Os índices ficam nos READMEs respectivos.

**Regra:** este `CLAUDE.md` é o **mapa**, não o **território**. Detalhes profundos vão para ADRs/Findings; o CLAUDE.md mantém apenas o resumo executivo e ponteiros. Quando houver conflito ou questão sobre "por que decidimos X", o ADR/Finding correspondente é a fonte canônica.

**Para o Claude Code (instruções de manutenção):**

- Antes de propor uma decisão metodológica nova, **leia** `docs/decisions/README.md` e os ADRs relevantes — pode haver decisão prévia que cobre o caso.
- Antes de fazer afirmação factual sobre o dataset, **leia** o Finding correspondente (ou crie um novo se não existir).
- Quando uma decisão antiga for revisitada, **não apague o ADR original** — crie um novo com `Status: Supersedes ADR-XXXX` e marque o antigo como `Superseded`.
- Adicione 1 linha no índice do README correspondente sempre que criar ADR ou Finding novo.
- Se um Finding sustentar uma decisão, **referencie-o explicitamente no ADR**.

## Trabalho relacionado / comparativo

Aluno incorporou em `CNN/Alzheimers disease dataset/Alzheimer.ipynb` um notebook prévio que será usado como comparativo no TCC. **Dataset diferente do nosso** (`uraninjo/augmented-alzheimer-mri-dataset`, sem subject IDs) e com **5 falhas metodológicas identificadas** — ver F-0005 para auditoria completa. Notebook é caso de estudo de "o que **não** fazer", justificativa empírica para nossas escolhas (split por paciente, transfer learning, métricas balanceadas, interpretabilidade).

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
