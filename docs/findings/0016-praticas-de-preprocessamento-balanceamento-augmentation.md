# F-0016 — Práticas de pré-processamento, balanceamento e augmentation (papers + best-practices externas)

- **Data:** 2026-05-17
- **Status:** Confirmed (auditoria dos 14 PDFs + 4 buscas na literatura externa com fontes)
- **Categoria:** metodologia

## Objetivo

Definir, com evidência, o pipeline de dados **correto e definitivo** antes de focar nos modelos: como pré-processar, balancear, aumentar e normalizar — respeitando split por paciente e explicabilidade — e responder à pergunta crítica: **balanceamento se aplica só no treino?** (resposta curta: **sim**, ver §3).

## 1. O que cada trabalho faz (auditoria interna)

| Trabalho | Pré-processamento | Balanceamento | Augmentation | Onde augmenta |
|---|---|---|---|---|
| **RanCom-ViT** (OASIS=nosso) | só resize 496×248→384×384 (usa Kaggle pronto) | **nenhum** (reconhece imbalance, mantém) | **nenhuma** (0 menções) | — |
| **Joint Transformer** (ADNI 3D) | MNI space, skull strip SPM12+CAT12 | não menciona | não menciona | — |
| **Leveraging Swin** (ADNI dMRI) | MP-PCA denoise, modelos microestruturais (DTI/NODDI), **sem registro a template** | balanced accuracy (métrica) | translações sincronizadas ±5 voxels (3D, antes de fatiar) | train |
| **Vision-ViT-CNN** (ADNI 3D) | **pipeline completo FSL/ANTs**: orient → registro MNI (FLIRT) → skull strip (BET) → bias field (N4) → enhancement (median filter + rescale + histogram equalization) | **weighted loss** (= nosso ADR-0007) | espacial p=1/3 (flip, affine, elastic) + intensidade p=1/4 (blur, motion, ghosting, noise) via TorchIO | **"training subsets" (explícito)** |
| **VGG-TSwinformer** (ADNI) | slices 112×112×3 | não menciona (pMCI/sMCI ~balanceado) | mirror flip + random rotation | train |

### Conclusões da auditoria interna

1. **O trabalho mais comparável ao nosso** (Vision-ViT-CNN: ADNI, T1, 3-class, split por paciente, 92,14%) usa **exatamente weighted loss** para balanceamento — validação independente do nosso ADR-0007.
2. **Augmentation é sempre train-only** quando os papers especificam (Vision-ViT-CNN: "training subsets"; os demais idem).
3. **Os trabalhos rigorosos usam augmentation; o que infla (RanCom-ViT) NÃO usa** — ele depende do leakage por slice para ter "dados suficientes". Sem leakage, augmentation é o substituto correto.
4. **Pré-processamento pesado (skull strip, registro MNI, bias field, enhancement) é padrão**, mas a maior parte já está feita no nosso dataset Kaggle (skull-stripped). O que **podemos** adicionar é enhancement de contraste (histogram equalization/CLAHE) e normalização — ver §4.
5. **Nenhum paper balanceia ou aumenta o test set.** Balanceamento é via loss (weighted) ou métrica (balanced acc); test reflete a distribuição real.

## 2. Best-practices externas (literatura geral, com fontes)

### Regra de ouro do balanceamento

> *"Any oversampling or undersampling technique (SMOTE, random over/undersampling) should only be applied to the training data and not the validation or test data."* — `imbalanced-learn`, common pitfalls.

> *"Applying oversampling before cross-validation will lead to high bias in radiomics"* — viés observado de **até 0,34 em AUC**. (Nature Scientific Reports, 2024)

Aplicar resampling ao dataset inteiro (ou antes do split/CV) **vaza** informação do test/val para o treino → métricas otimistas e irreais. O correto é resamplear **dentro de cada fold, apenas na porção de treino**.

### Augmentation

> *"Apply data augmentation only to the training samples... split your data first, then augment only the training set."* — augmentar antes de separar val/test causa leakage. TTA (test-time augmentation) é exceção avançada que **não** usaremos.

### Normalização de intensidade MRI

> Z-score, Nyúl e WhiteStripe são os métodos populares; normalização + enhancement de contraste melhoram acurácia (até **+4,5%** com preproc mais profundo, especialmente em ViT). (ScienceDirect; Nature Sci Reports)

**Tensão para o nosso caso:** Nyúl/WhiteStripe dependem da distribuição de intensidade do MRI bruto — nosso dataset já é **JPG** (escala de intensidade original perdida na conversão), então esses métodos não se aplicam bem. Z-score per-image é viável, **mas** usamos pesos ImageNet (transfer learning), cuja prática padrão é normalização ImageNet. Decisão em ADR-0011.

### 2.5D (slices vizinhos como canais)

> *"Three consecutive slices (current + adjacent neighbors) concatenated along the channel dimension, forming a three-channel input analogous to RGB... balancing spatial context of 3D models with computational efficiency of 2D... successfully addressed challenges of small-scale datasets."*

Técnica reconhecida que ataca diretamente nosso ponto fraco (RGB sintético R=G=B desperdiça 2 canais). Candidata a melhoria (V3) — ADR-0011.

## 3. RESPOSTA à pergunta central: balanceamento só no treino?

**SIM, inequivocamente.** Confirmado por (a) todos os papers do projeto que tratam imbalance (Vision-ViT-CNN: weighted loss no treino), (b) a biblioteca canônica `imbalanced-learn`, (c) Nature Sci Reports (viés de até 0,34 AUC se violado).

| Conjunto | Balanceamento? | O que se faz |
|---|---|---|
| **Train** | ✅ Sim | weighted loss (nosso ADR-0007) OU oversampling/sampler **dentro do fold** |
| **Validation** | ❌ Nunca | dados intactos; monitora métricas balanceadas |
| **Test** | ❌ **Nunca** | dados intactos = distribuição real do mundo; reporta balanced acc / macro-F1 / AUC |

**Por que no teste não faz sentido (sua intuição estava certa):** o test set deve representar a realidade clínica, onde Non Demented é maioria. Balancear o teste artificialmente mediria desempenho num mundo que não existe e inflaria/distorceria as métricas. O desbalanceamento no teste é tratado **na métrica** (balanced accuracy dá peso igual a cada classe independente do tamanho), **não nos dados**.

**Conformidade do nosso código (verificada):**
- `build_dataloaders`: augmentation só no `train` (`train_augment_strength`), val/test `"off"`. ✓
- `compute_class_weights`: pesos calculados sobre o **train loader**, não global. ✓
- `evaluate`: métricas (balanced_acc, macro-F1, AUC) computadas sobre `y_true/y_pred` **puros**; a loss ponderada só entra no `val_loss` de monitoramento, nunca nas métricas. ✓
- Split por paciente (ADR-0002) garante que o resampling/weighting do treino não vaza para test. ✓

## Implicação

Pipeline de dados consolidado em **ADR-0011**. Resumo das decisões:
- **Manter**: resize 224 squash + ImageNet norm (ADR-0004); weighted CE train-only (ADR-0007); augmentation strong train-only (ADR-0006/V2); métricas balanceadas (ADR-0005); split por paciente (ADR-0002).
- **Adotar como ablações opcionais** (depois do baseline V2): enhancement CLAHE; augmentation de intensidade (blur/noise via TorchIO); pipeline 2.5D (slices vizinhos).
- **Rejeitar**: oversampling/SMOTE de slices (poucos pacientes → não cria diversidade real; e complexidade de aplicar dentro do fold); qualquer balanceamento/augmentation em val/test; Nyúl/WhiteStripe (incompatível com JPG).

## Notas / armadilhas

- **2.5D no nosso dataset**: o índice de slice (`_NNN`) permite pegar vizinhos (n-1, n, n+1) do mesmo `(subject, session, mpr)`. Cuidado nas bordas (slice 100 e 160) — replicar a borda ou pular. Detalhar quando implementar V3.
- **CLAHE/histogram equalization** pode realçar ruído de fundo (área preta do skull strip). Aplicar com `clip_limit` moderado e validar visualmente (lição F-0003) antes de adotar.
- **Não migrar para z-score** sem re-treinar do zero: muda a distribuição de entrada e quebra compatibilidade com pesos ImageNet pré-treinados.
- Fontes web são de maio/2026; conferir DOIs antes de citar formalmente no TCC.

## Fontes

- imbalanced-learn — common pitfalls: https://imbalanced-learn.org/stable/common_pitfalls.html
- "Applying oversampling before cross-validation will lead to high bias in radiomics" (Nature Sci Reports 2024): https://www.nature.com/articles/s41598-024-62585-z
- "MR image normalization dilemma and the accuracy of brain tumor classification" (ScienceDirect): https://www.sciencedirect.com/science/article/pii/S1687850722071758
- "Standardization of brain MR images across machines and protocols" (Nature Sci Reports): https://www.nature.com/articles/s41598-020-69298-z
- 2.5D transfer deep learning (Quantitative Imaging in Medicine and Surgery / PMC): https://pmc.ncbi.nlm.nih.gov/articles/PMC10784073/
- Data augmentation practice (Baeldung CS): https://www.baeldung.com/cs/ml-data-augmentation
