# F-0014 — Auditoria-mestre: split por paciente nos trabalhos relacionados (alicerce do TCC)

- **Data:** 2026-05-17
- **Status:** Confirmed (leitura integral dos 14 PDFs + auditoria do dataset uraninjo com scripts reproduzíveis)
- **Categoria:** trabalho-relacionado

## Objetivo

Determinar, com evidência literal, **quais trabalhos relacionados usam split por paciente, quais não usam, e quais não deixam explícito** — e provar se o dataset usado pela CNN comparativa (uraninjo) permite ou não identificar paciente. Este finding é alicerce metodológico: sustenta por que nossas métricas (mais baixas) são mais honestas que as de trabalhos que reportam 99%+.

## Metodologia da auditoria

1. Texto dos 14 PDFs em `docs/Papers/Trabalhos Relacionados/` extraído via `src/data/extract_paper_texts.py` (pypdf) e auditado seção a seção (dataset / método / split).
2. Dataset uraninjo auditado via `src/data/audit_cnn_dataset.py`: nomenclatura, EXIF, dimensões, duplicatas (hash MD5) intra e entre classes, cópias Original↔Augmented; + inspeção visual.
3. Cada conclusão abaixo vem acompanhada de **citação literal** (entre aspas) localizada no PDF.

## Quadro-mestre

| Trabalho | Dataset | Unidade de split | Split por paciente? | Evidência (citação literal) |
|---|---|---|---|---|
| **RanCom-ViT** (Lu, Zhang, Yao 2025) | **OASIS-Kaggle = O NOSSO** | slice (imagem 2D) | ❌ **Não** | *"we split the dataset randomly into 80% for training and 20% for testing... we stratified the dataset by class during the split"* |
| **Joint Transformer** (Scientific Reports 2024) | ADNI 3D (longitudinal) | scan/volume 3D | ⚠️ **Parcial** | *"A random split approach was used, allocating 60% for training and 20% for testing and validation, resulting in sample sizes of (211, 70, 70)"* (samples = scans, Table 1: 351 e 2294 scans) — sem agrupamento por sujeito num dataset longitudinal |
| **Leveraging Swin** (IEEE TBME) | ADNI dMRI | **sujeito** | ✅ **Sim (explícito)** | *"15% of unique subjects were randomly assigned to a fixed hold-out test set... 5-fold grouped and stratified cross-validation. This strategy ensured that all data from any given subject appeared entirely within either the training or validation set"* |
| **VGG-TSwinformer** (longitudinal MCI) | ADNI | **sujeito** | ✅ **Sim (explícito)** | *"multiple samples from the same subject are only allowed to exist in one of the training, validation and test subsets. This is very important because samples belonging to the same subject have a [correlation]"* |
| **Vision-ViT-CNN** (Frontiers in Neurology) | ADNI (818 participantes) | **sujeito** | ✅ **Sim (explícito)** | *"A subject-level split strategy was adopted during the cross-validation process"* |
| **detection_of_Alzheimer** (review/meta-análise QUADAS-2) | 16 estudos | — (review) | meta-evidência | *"About 31% of the outcomes in the included studies lacked details on patient selection, leading to a high risk of bias"* |
| **Notebook CNN do aluno** | uraninjo | imagem 2D | ❌ **Impossível** (ver abaixo) | `flow_from_directory(validation_split=0.2)` sem subject ID disponível |

**Papers de arquitetura base** — ViT (Dosovitskiy), Swin (Liu), ResNet (He), Transformer (Vaswani), Grad-CAM, Chefer, Attention Rollout (Abnar), survey Shamshad — **não fazem classificação de Alzheimer**; treinam em ImageNet/COCO/NLP, onde "split por paciente" não se aplica. Não entram na comparação metodológica de leakage, mas são as fontes das arquiteturas/técnicas que usamos.

## Detalhamento crítico

### 1. RanCom-ViT — mesmo dataset que o nosso, split por slice (já em F-0008, reconfirmado)

Linhas 253/456/462 do PDF: *"461 patients"*, *"we split the dataset randomly into 80% for training and 20% for testing"*, *"we stratified the dataset by class during the split"*. **Split aleatório por slice, estratificado por classe, não por paciente.** Como cada paciente tem ~244 slices, isso garante que slices do mesmo paciente caem em train E test → leakage. Reporta 99,54% acurácia. Também descreve o plano de corte erradamente como "sagittal" (é axial — F-0008). **É o comparativo mais importante: mesmo dataset, metodologia oposta à nossa.**

### 2. Joint Transformer — split por volume, mas leakage longitudinal não controlado

A unidade de split são **scans 3D** (Table 1: 351 scans no ADNI 3Yr; 2294 no 1Yr), não slices — então **não há leakage por slice**. Porém, ADNI é longitudinal: *"subjects who had scans taken at screening and at 6- and 12-month visits"*. O "random split" das amostras (scans) **não menciona agrupamento por sujeito** — então scans diferentes do mesmo paciente (visitas distintas) podem cair em train e test. É **melhor que split por slice, mas não é split por paciente rigoroso**. Classificar como "zona cinzenta / não explícito quanto a sujeito".

### 3-5. Leveraging Swin, VGG-TSwinformer, Vision-ViT-CNN — split por paciente explícito

Os três **afirmam textualmente** fazer split no nível do sujeito. Mais ainda:
- **Leveraging Swin** disponibiliza os subject identifiers e **critica explicitamente quem não faz**: *"Recent transformer-based pipelines on sMRI can also reach very high accuracies under random splits, underscoring how evaluation design can inflate headline metrics."* — esta frase é sustentação direta da nossa tese.
- **VGG-TSwinformer** justifica a importância (correlação intra-sujeito enviesa o modelo) — exatamente nosso argumento.

### 6. Meta-análise — 31% dos estudos não relatam seleção de paciente

A review `detection_of_Alzheimer` (QUADAS-2, 16 estudos) encontrou que **~31% dos estudos têm alto risco de viés por falta de detalhe na seleção de pacientes**. Evidência META de que a inconsistência de relato metodológico (incluindo split) é problema reconhecido na área.

## O dataset uraninjo (CNN comparativa) — split por paciente é IMPOSSÍVEL

Auditoria via `src/data/audit_cnn_dataset.py`. Cinco evidências independentes de que **não há como identificar o paciente de origem**:

1. **Nomenclatura sem subject ID.** Dois padrões coexistem em cada classe:
   - `# (#).jpg` (ex: `26 (19).jpg`, `26 (100).jpg`)
   - `<classe>Dem#.jpg` (ex: `mildDem717.jpg`, `nonDem2560.jpg`) — numeração **sequencial por classe**, claramente atribuída na curadoria.
   Nenhum dos dois codifica paciente.

2. **O mesmo prefixo numérico aparece em classes diferentes.** O prefixo `26` aparece em `MildDemented` (`26 (19)`), `NonDemented` (`26 (100)`) e `VeryMildDemented` (`26 (44)`). Se `26` fosse um paciente, ele estaria simultaneamente em três diagnósticos distintos — **clinicamente impossível**. Logo, o prefixo **não é** identificador de paciente.

3. **EXIF vazio.** `img._getexif()` retorna `None` em todas as amostras — zero metadados de origem (sem DICOM tags, sem data, sem ID).

4. **Imagens 176×208 grayscale (mode=L), sem cabeçalho clínico.** Diferente do nosso OASIS-Kaggle (496×248 RGB). JPG simples, sem qualquer estrutura que carregue proveniência.

5. **Sem duplicatas exatas que permitam reconstruir grupos.** Cada imagem é única por hash (0 duplicatas intra-classe; 0 imagens idênticas entre classes). Não há como agrupar por identidade de conteúdo.

**Conclusão:** no dataset uraninjo, a informação de paciente foi **destruída na curadoria**. Qualquer trabalho que o use — incluindo o notebook CNN do aluno — **só pode fazer split por imagem**, e **não há nem como auditar** se há leakage de paciente. Isso é mais grave que o RanCom-ViT (onde o subject ID existe no nome e o split por slice é uma escolha): aqui a escolha rigorosa nem é possível.

Nota honesta sobre cópias Augmented↔Original: `audit_cnn_dataset.py` encontrou **0 cópias byte-a-byte** entre `AugmentedAlzheimerDataset` e `OriginalDataset` (hash difere porque augmentation altera pixels). Portanto **não podemos afirmar "cópia exata"** como prova de leakage. O argumento de leakage do notebook do aluno sustenta-se pela **impossibilidade de split por paciente** (acima), não por duplicação literal — ver F-0005 revisado.

## Implicações para o TCC

1. **A tese é nuançada e defensável**, não um "todos erram":
   - Os trabalhos **mais rigorosos** da área (Leveraging Swin, VGG-TSwinformer, Vision-ViT-CNN) **fazem split por paciente** e um deles **critica explicitamente** quem usa random split.
   - Mas no **mesmo dataset que usamos** (OASIS-Kaggle), o trabalho de referência (RanCom-ViT) faz **split por slice** e reporta 99,54% — provável artefato.
   - Datasets populares como o **uraninjo tornam o split por paciente impossível**, perpetuando métricas não auditáveis.
2. **Nosso diferencial**: split por paciente rigoroso (ADR-0002) no OASIS-Kaggle, alinhado às melhores práticas da área, contra o RanCom-ViT. Métricas mais baixas = mais honestas.
3. **Frase-âncora para a escrita** (parafrasear, não copiar): a própria literatura rigorosa reconhece que "evaluation design can inflate headline metrics" sob random splits (Leveraging Swin) — usamos isso para enquadrar o contraste com RanCom-ViT.
4. **Material direto para a seção "Trabalho Relacionado"**: o quadro-mestre acima vira tabela do TCC.

## Notas / armadilhas

- Citações foram extraídas de texto via pypdf; **conferir contra o PDF original antes de citar formalmente no TCC** (paginação e quebras podem distorcer). Os números de página/linha referem-se aos `.txt` gerados (regeneráveis).
- "Subject-level split" afirmado por um paper ainda pode ter falhas não relatadas (não temos o código). Tratamos a afirmação textual como evidência do *intento metodológico*, que é o que se compara.
- Joint Transformer fica em "zona cinzenta" — não afirmar que "faz leakage", mas que "não controla explicitamente leakage longitudinal por sujeito". Rigor na acusação.
- A origem do dataset uraninjo é disputada na comunidade Kaggle (alguns dizem derivar de OASIS). **Irrelevante** para nossa conclusão: independente da origem, o subject ID não está disponível ali.
