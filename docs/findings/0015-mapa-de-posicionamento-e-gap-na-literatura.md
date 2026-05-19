# F-0015 — Mapa de posicionamento na literatura: métricas, oponente direto e o gap do TCC

- **Data:** 2026-05-17
- **Status:** Confirmed (entre os 14 PDFs do projeto; ver "Limite de escopo")
- **Categoria:** trabalho-relacionado

## Objetivo

Responder, com números auditados, a três perguntas estratégicas do TCC:
1. Quais métricas cada trabalho alcançou, e em que dataset?
2. Existe um "oponente direto" (mesmo dataset + mesma arquitetura + mesmo split)?
3. Alguém faz **ViT + OASIS + split por paciente**? Se não, isso é o nosso gap/contribuição.

Métricas extraídas dos PDFs via `src/data/extract_paper_texts.py` + grep (citações conferíveis nos `.txt` regeneráveis).

## Matriz-mestre de posicionamento

| Trabalho | Dataset | Modalidade | Arquitetura | Split | Tarefa | Métrica headline | Métrica reportada |
|---|---|---|---|---|---|---|---|
| **RanCom-ViT** (Lu 2025) | **OASIS-Kaggle = NOSSO** | T1 2D | ViT + token compression | ❌ slice | 4-class | **99,54%** | accuracy (não balanceada) |
| **Joint Transformer** (2024) | ADNI 3D | T1 3D→2D | ViT + Time-Series Transformer | ⚠️ volume (longitudinal não controlado) | binário / 3-class | **99,05% / 99,01%** | accuracy |
| **Leveraging Swin** | ADNI | **dMRI (DTI/NODDI)** | Swin + LoRA | ✅ sujeito | binário (AD/CN; MCI/CN) | **95,2% / 83,8%** | **balanced accuracy** |
| **Vision-ViT-CNN** | ADNI | T1 3D | ViT + CNN | ✅ sujeito | 3-class (NC/MCI/AD) | **92,14%** | accuracy |
| **VGG-TSwinformer** | ADNI | T1 longitudinal | VGG-16 + Swin temporal | ✅ sujeito | pMCI vs sMCI | **77,2%** | accuracy |
| **detection_of_Alzheimer** | meta (16 estudos) | vários | review ViT | — | — | — | QUADAS-2 |
| **NOSSO TCC (baseline V1)** | **OASIS-Kaggle** | T1 2D | ResNet-50 (baseline) | ✅ **sujeito** | 3-class | bal_acc **0,623** | bal_acc + macro-F1 + AUC |
| **NOSSO TCC (alvo)** | OASIS-Kaggle | T1 2D | **ViT-Base + Swin-T** | ✅ sujeito | 3-class | a obter | bal_acc + macro-F1 + AUC + McNemar |

Baseline V1 detalhado em F-0013. Referência cruzada de split em F-0014.

## Leitura da matriz — três conclusões

### 1. Não há "oponente direto" para superar em número — e isso é metodologicamente correto

Comparar accuracy entre **datasets diferentes** (OASIS vs ADNI), **modalidades diferentes** (T1 vs dMRI), **tarefas diferentes** (4-class vs binário vs pMCI/sMCI) e **splits diferentes** (slice vs volume vs sujeito) é **inválido**. Os 99% do RanCom-ViT e os 95,2% do Leveraging Swin **não são comparáveis entre si nem com o nosso** — medem coisas diferentes em condições diferentes.

Portanto, **não temos um "boss" para vencer em número**. Tentar "bater 99,54%" seria cair na mesma armadilha que critica-se.

### 2. O oponente é METODOLÓGICO, e tem nome: RanCom-ViT

É o **único** trabalho de classificação no **exato mesmo dataset** (OASIS-Kaggle `ninadaithal/imagesoasis`) e na **mesma família de arquitetura** (ViT). A diferença é o split: ele por slice (99,54% inflado), nós por paciente. **A comparação central do TCC é: mesmo dado, metodologia oposta, o que muda no resultado.** Nosso número será mais baixo e mais honesto — e isso é o ponto.

### 3. O gap: ViT + OASIS + split por paciente não existe entre os trabalhos auditados

| Combinação | Existe? | Quem |
|---|---|---|
| ViT + OASIS + split por **slice** | ✅ | RanCom-ViT |
| ViT/Swin + ADNI + split por **paciente** | ✅ | Leveraging Swin, Vision-ViT-CNN, VGG-TSwinformer |
| **ViT + OASIS + split por paciente** | ❌ **vazio** | **— (nosso TCC)** |

Os que fazem split por paciente migraram para ADNI (que tem subject IDs limpos e modalidades ricas como dMRI). Quem ficou no OASIS-Kaggle 2D fez split por slice. **A interseção "rigor metodológico (split por paciente) + dataset OASIS-Kaggle 2D + Vision Transformer" está vazia — é onde nosso TCC se posiciona.**

## Papel de cada trabalho na argumentação do TCC

| Trabalho | Papel na escrita | Onde entra |
|---|---|---|
| **RanCom-ViT** (F-0008) | **Contraste metodológico central** — mesmo dataset, split por slice, 99,54%. Mostra o problema que resolvemos. | Introdução (motivação) + Trabalho Relacionado + Discussão (comparação direta) |
| **CNN do aluno / uraninjo** (F-0005, F-0014) | **Segundo contraste** — dataset sem subject ID, split por paciente impossível. | Trabalho Relacionado + Discussão (limitações de datasets populares) |
| **Leveraging Swin** | **Referência de boa prática + frase-âncora** ("evaluation design can inflate headline metrics") + benchmark de teto realista com split honesto (bal_acc 90-95% binário dMRI). | Metodologia (justifica split por paciente) + Discussão |
| **Vision-ViT-CNN** | **Boa prática (subject-level) + benchmark 3-class** (92% em ADNI 3D — mas T1 estrutural como o nosso). | Metodologia + Discussão (3 classes é viável com rigor) |
| **VGG-TSwinformer** | **Boa prática (split por sujeito) + tarefa difícil** (pMCI/sMCI 77% mostra que rigor → números modestos). | Metodologia + Discussão (números honestos são mais baixos) |
| **Joint Transformer** | **Zona cinzenta** — split por volume sem controle longitudinal; exemplo de como ADNI longitudinal pode inflar mesmo "sem split por slice". | Trabalho Relacionado (nuance de tipos de leakage) |
| **detection_of_Alzheimer** (review) | **Meta-evidência** — 31% dos estudos sem detalhe de seleção de paciente (alto risco de viés). | Introdução (problema sistêmico da área) |
| **ViT (Dosovitskiy), Swin (Liu), ResNet (He), Transformer (Vaswani)** | **Fontes das arquiteturas** que usamos. | Fundamentação teórica |
| **Grad-CAM (Selvaraju), Attention Rollout (Abnar), Chefer** | **Fontes das técnicas de interpretabilidade**. | Fundamentação + Interpretabilidade |
| **Survey Shamshad (Transformers in Medical)** | **Panorama geral** de transformers em imagem médica. | Fundamentação / Introdução |

## Implicações para o TCC

1. **Contribuição declarável:** primeira aplicação (entre os trabalhos auditados) de Vision Transformers ao OASIS-1 (2D Kaggle) com **split por paciente rigoroso + métricas balanceadas + interpretabilidade**, contrastando diretamente com o estado da arte do mesmo dataset (RanCom-ViT) que usa split por slice.
2. **Como reportar números:** sempre bal_acc + macro-F1 + AUC (nunca accuracy isolada); comparar **contra nós mesmos** (ablações com/sem peso, ResNet vs ViT vs Swin via McNemar) e **contra o RanCom-ViT no plano metodológico** (não no plano de número absoluto).
3. **Expectativa calibrada:** com split por paciente honesto em T1 2D, o teto realista é provavelmente bal_acc 0,65-0,80 (comparar: Vision-ViT-CNN 92% em ADNI 3D 3-class é com volume completo + modalidade mais rica). Não nos assustar com números abaixo de 99%.

## Limite de escopo (honestidade)

- Este mapa cobre **os 14 PDFs do projeto**, escolhidos pelo aluno. **Não é uma busca sistemática** da literatura mundial. Afirmar "ninguém no mundo fez ViT+OASIS+split por paciente" exige busca formal.
- **Próximo passo recomendado** (antes da escrita final do Trabalho Relacionado): busca sistemática em Google Scholar / PubMed / IEEE Xplore com termos como `("vision transformer" OR ViT OR Swin) AND OASIS AND ("subject-level" OR "patient-level" OR "subject-wise" split)`. Se confirmar o vazio, a contribuição fica mais forte; se achar algo, incorporamos como comparativo.
- Métricas foram lidas via pypdf; **conferir no PDF original antes de citar** no texto final (paginação/quebras podem distorcer).
