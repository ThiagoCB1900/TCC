# FAQ conceitual — defesa de banca

Respostas curtas e defensáveis para as perguntas metodológicas mais prováveis na banca. Cada uma aponta para o ADR/Finding que a fundamenta. **Não é o texto do TCC** — é um colante de defesa oral; o texto formal expande estes pontos.

> Convenção: "resposta de 30 segundos" = o que dizer primeiro; "se aprofundarem" = munição extra.

---

## 1. Vocês fazem balanceamento de classes?

**Resposta de 30s:** Sim. Tratamos o desbalanceamento com **perda ponderada por classe** (weighted cross-entropy), não com reamostragem dos dados. "Balanceamento" tem dois tipos: por **dados** (oversampling/SMOTE) e por **custo** (pesar mais os erros das classes raras). Usamos o segundo.

**Se aprofundarem:**
- Precisamos: o desbalanceamento é ~12× entre classes (pacientes) e 138× (slices). Sem tratar, o modelo colapsa na classe majoritária.
- Provamos empiricamente (baseline V1, F-0013): sem peso → balanced_acc 0,50 e F1 da minoritária 0,17; com peso → 0,62 e 0,40 (+135% na minoritária).
- Não usamos oversampling porque há só 17 pacientes da classe minoritária no treino — repetir fatias deles é falsa diversidade, não regularização.
- Fundamentação: ADR-0007, F-0016. Validado externamente: Vision-ViT-CNN (trabalho mais comparável) usa exatamente weighted loss.

## 2. A augmentation no treino não causa data leakage?

**Resposta de 30s:** Não, porque é aplicada **só ao conjunto de treino, depois do split por paciente**. A augmentation gera variações de imagens que já são de treino — nenhuma informação de validação/teste entra no processo.

**Se aprofundarem:**
- Leakage seria aumentar **antes** de dividir os dados (versões da mesma imagem cairiam em treino e teste) — o que evitamos deliberadamente.
- Val/test ficam intactos (sem augmentation), refletindo a distribuição real.
- Esse erro (augmentar antes de separar) é uma das falhas do notebook CNN comparativo (F-0005).
- Defesa dupla: split por paciente (1ª linha) + augmentation train-only (2ª linha). Fundamentação: ADR-0006, ADR-0011, F-0016.

## 3. Por que não reportam acurácia como métrica principal?

**Resposta de 30s:** Porque o dataset é fortemente desbalanceado — um classificador trivial que sempre prediz "Non Demented" já alcança ~78% de acurácia. Usamos **balanced accuracy, macro-F1 e AUC**, que dão peso igual a cada classe.

**Se aprofundarem:**
- Acurácia bruta esconde o fracasso nas classes minoritárias, justamente as clinicamente críticas (estágios de demência).
- Reportamos acurácia apenas como métrica secundária, para comparabilidade com a literatura.
- Fundamentação: ADR-0005.

## 4. Por que 3 classes (e não as 4 originais ou binário)?

**Resposta de 30s:** A classe "Moderate Dementia" tem só **2 pacientes** no dataset inteiro. Split por paciente em 4 classes deixaria o teste com 0–1 paciente nessa classe — estatisticamente sem sentido. Fundimos Mild+Moderate, mantendo granularidade clínica onde os dados permitem.

**Se aprofundarem:**
- O esquema fica: Non Demented (266 pac.) / Very Mild (58) / Mild+Moderate (23).
- O `Dataset` também suporta o esquema binário como fallback configurável.
- Fundamentação: ADR-0001, F-0002.

## 5. Por que split por paciente, e não aleatório?

**Resposta de 30s:** Cada paciente contribui ~244 fatias. Um split aleatório por fatia coloca o mesmo paciente em treino e teste, fazendo o modelo decorar o indivíduo em vez de aprender a doença — inflando a métrica artificialmente.

**Se aprofundarem:**
- É a regra crítica nº 1 do projeto. Garantimos conjuntos disjuntos de subject IDs (validação automática).
- A literatura rigorosa concorda: Leveraging Swin, VGG-TSwinformer e Vision-ViT-CNN fazem split por paciente; o Leveraging Swin chega a afirmar que "evaluation design can inflate headline metrics" sob random splits.
- O RanCom-ViT, no mesmo dataset que o nosso, faz split por fatia e reporta 99,54% — provável artefato.
- Fundamentação: ADR-0002, F-0008, F-0014.

## 6. Vocês reportam métricas mais baixas que a literatura (99%). Por quê?

**Resposta de 30s:** Porque nossas métricas são **honestas**. Os 99%+ da literatura no mesmo dataset vêm de split por fatia (leakage). Comparar contra esses números seria comparar maçãs com laranjas. Comparamos contra nós mesmos (ablações) e contrastamos a metodologia.

**Se aprofundarem:**
- Comparar accuracy entre datasets/modalidades/splits diferentes é metodologicamente inválido (F-0015).
- O "oponente" não é para vencer em número — é o contraste metodológico com o RanCom-ViT (mesmo dataset, split por fatia).
- Tarefas mais difíceis com split rigoroso reportam números modestos: VGG-TSwinformer 77,2% (pMCI/sMCI, split por paciente).
- Fundamentação: F-0008, F-0013, F-0015.

## 7. Por que Vision Transformers, e não só CNN?

**Resposta de 30s:** ViTs capturam relações globais via auto-atenção e, com pesos pré-treinados, competem ou superam CNNs em imagem médica. Mantemos a ResNet-50 como baseline obrigatório para comparação justa (mesma metodologia).

**Se aprofundarem:**
- Comparamos ResNet-50 (baseline) vs ViT-Base vs Swin-T no mesmo split/test, com McNemar para significância estatística.
- A interpretabilidade do ViT (Attention Rollout) é um diferencial clínico.
- Fundamentação: plano do TCC; arquiteturas de Dosovitskiy (ViT), Liu (Swin), He (ResNet).

## 8. Por que OASIS e não ADNI (mais usado)?

**Resposta de 30s:** OASIS-1 é público e acessível, e — crucialmente — **nenhum trabalho auditado combina ViT + OASIS + split por paciente**. Os que fazem split por paciente usam ADNI; quem usa OASIS faz split por fatia. Ocupamos esse espaço vazio.

**Se aprofundarem:**
- É o gap/contribuição: rigor metodológico (split por paciente) aplicado ao OASIS-Kaggle 2D com ViT.
- Limite honesto: gap confirmado entre os 14 papers do projeto; recomendamos busca sistemática antes da escrita final.
- Fundamentação: F-0015.

## 9. Por que resize "squash" 224×224 e RGB sintético (R=G=B)?

**Resposta de 30s:** As imagens são fatias axiais 2D alongadas (496×248). O resize para 224×224 é o padrão de entrada dos modelos pré-treinados em ImageNet. Mantemos 3 canais (idênticos) por compatibilidade com esses pesos.

**Se aprofundarem:**
- Squash é padrão em ViT médico; o modelo aprende a invariância à deformação consistente.
- RGB sintético desperdiça 2 canais — por isso o pipeline **2.5D** (3 fatias vizinhas como canais) é nossa principal ablação de melhoria reservada.
- Fundamentação: ADR-0004, ADR-0011, F-0003.

## 10. Como garantem que o pré-processamento não introduz viés?

**Resposta de 30s:** O dataset Kaggle já vem skull-stripped e fatiado. Aplicamos apenas resize + normalização ImageNet (determinísticos) + augmentation só no treino. Balanceamento e augmentation nunca tocam val/test.

**Se aprofundarem:**
- Validação visual de cada etapa do pipeline antes de treinar (lição de um erro anterior — F-0003).
- Regras de ouro: split por paciente; aug/balance só no treino; teste reflete distribuição real (ADR-0011).
- Fundamentação: ADR-0011, F-0016.

---

## Limitações que assumimos proativamente (mostrar maturidade)

- Trabalhamos com 347 dos 416 pacientes do OASIS-1 oficial (a versão Kaggle excluiu pacientes sem CDR válido) — F-0002.
- Moderate Dementia tem só 2 pacientes → fundida com Mild — ADR-0001.
- Imagens são JPG (intensidade MRI original perdida na conversão) → não usamos normalização específica de MRI (Nyúl/WhiteStripe) — F-0016.
- Gap na literatura confirmado entre os 14 papers do projeto, não por busca sistemática mundial — F-0015.
