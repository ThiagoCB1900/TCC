# F-0008 — RanCom-ViT (Lu et al. 2025) usa o mesmo dataset com metodologia comprometida

- **Data:** 2026-05-10
- **Status:** Confirmed (extração literal do PDF)
- **Categoria:** trabalho-relacionado

## O que descobrimos

O paper **"An efficient vision transformer for Alzheimer's disease classification using magnetic resonance images"** (Lu, Zhang, Yao — *Biomedical Signal Processing and Control*, 101, 2025, 107263) propõe o modelo **RanCom-ViT** e o avalia **exatamente no mesmo dataset Kaggle que o nosso TCC** (`ninadaithal/imagesoasis`, 86.437 slices, distribuição idêntica por classe). Reporta **acurácia de 99,54%**.

A metodologia tem **três falhas relevantes** que tornam o resultado incomparável a um pipeline rigoroso:

### Falha 1 — Split aleatório por slice, não por paciente

Trecho literal (página 6 do PDF):

> "we split the dataset **randomly** into 80 % for training and 20 % for testing. […] To ensure an equal and random distribution of each dementia stage […], we **stratified the dataset by class** during the split."

Não há agrupamento por subject ID. Como cada paciente contribui em média ~244 slices (4 mpr × 61 slices) e nossos dados mostram que existem 347 pacientes para 86.437 slices, um split aleatório **garante que praticamente todo paciente tem slices em train e test simultaneamente**. O modelo aprende características anatômicas individuais dos pacientes, não a doença propriamente. Esta é a regra crítica nº 1 do nosso CLAUDE.md, violada explicitamente.

Para a classe Moderate Dementia (apenas 2 pacientes no dataset, ver F-0002): o modelo viu múltiplas aquisições mpr e múltiplos slices de **ambos os pacientes** durante o treino, sobrando ~98 slices dos mesmos 2 pacientes para o "test". O paper reporta 100% de precision e specificity para Moderate — resultado incompatível com generalização real.

### Falha 2 — Descrição incorreta do plano de slicing

Trecho literal (página 3):

> "Each 3D MRI volume is **split into 256 2D scans along the sagittal view**, and only the scans ranging from the 100th to the 160th are added to the dataset."

Os slices do dataset Kaggle são **axiais**, não sagitais. Confirmado por:

1. Inspeção visual direta de `Data/Non Demented/OAS1_0001_MR1_mpr-1_130.jpg`: os dois hemisférios cerebrais aparecem **lado a lado**, ventrículos laterais simétricos, globos oculares anteriores — anatomia **incompatível com vista sagital** (que mostraria perfil de **um único hemisfério**).
2. Descrição do próprio Kaggle: "*sliced along the **z-axis** into 256 pieces*" — slicing ao longo do eixo z gera fatias axiais (planos perpendiculares ao eixo).

Erro descritivo num trecho metodológico de um artigo publicado é indicador de descuido editorial.

### Falha 3 — Métricas enviesadas pelo desbalanceamento

Métricas reportadas (página 6, Tabela 2): precision, sensitivity, specificity, F1 score, accuracy — todas **per-class** e **average simples**, sem `balanced accuracy`, `macro-F1` ponderado de forma controlada, ou matriz de confusão normalizada por linha. Em dataset onde Non Demented é 77,8% das amostras, accuracy crua tem valor informativo limitado.

## Evidência

- PDF: `docs/Papers/Trabalhos Relacionados/An_efficient_vision_transformer_for_Alzheimer's_disease_classification.pdf`
- Texto extraído em UTF-8: `docs/findings/_rancom_vit_text.txt` (gerado por `src/data/_extract_paper_text.py`)
- Trechos citados em linhas 253-263 (descrição do dataset), 446-469 (split), 489-498 (métricas).
- Inspeção visual: `Data/Non Demented/OAS1_0001_MR1_mpr-1_130.jpg` — clara vista axial.
- Distribuição nas tabelas do paper bate exatamente com `results/eda/summary.json` deste projeto.

## Implicação

Este achado é **central para a justificativa metodológica do TCC**:

1. **Trabalho relacionado direto.** RanCom-ViT é o estado da arte publicado mais recente no exato dataset que usaremos. Não pode ser ignorado.
2. **Justificativa empírica para split por paciente (ADR-0002).** Não estamos sendo "exigentes demais" — estamos seguindo a prática rigorosa que difere do que foi publicado.
3. **Experimento de ablação valioso (já listado na ordem de corte do CLAUDE.md):** treinar a CNN do notebook comparativo (F-0005) e/ou um ViT pré-treinado simples no nosso dataset com metodologia rigorosa, e mostrar a queda esperada em relação aos 99,54% do RanCom-ViT.
4. **Não almejar superar 99,54%.** Esse número é artefato metodológico. Nossa baseline será comparada com **ela mesma sob diferentes esquemas** (com vs sem peso, com vs sem augmentation), não com a literatura comprometida.
5. **Cuidado com expectativas de orientador / banca.** Se houver pressão para "bater o RanCom-ViT", explicar com este finding na mão por que isso é maçã-com-laranja.

## Notas / armadilhas

- **Risco de leitura superficial:** RanCom-ViT é um paper de 2025 publicado em *Biomedical Signal Processing and Control* (Q1, IF ~5). Quem ler o resumo sem auditar a metodologia pode usar os 99,54% como referência. Justifica seção dedicada no Trabalho Relacionado.
- **Outros papers no mesmo dataset Kaggle podem ter os mesmos problemas.** Auditar antes de citar como referência. Procurar especificamente por "split by patient", "subject-level", "subject-wise" nas seções de método.
- **Reproducibilidade do paper:** sem código público mencionado no extrato. Não podemos rodar o código deles para validar.
- **Eventual contato:** se o orientador quiser, pode-se tentar contatar os autores para confirmação, mas não é necessário para o argumento.
