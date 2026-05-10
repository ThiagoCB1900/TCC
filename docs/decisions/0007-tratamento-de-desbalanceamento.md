# ADR-0007 — Tratamento de desbalanceamento: Weighted CrossEntropy "balanced", com ablação sem peso

- **Data:** 2026-05-10
- **Status:** Accepted
- **Decisores:** Thiago, Claude

## Contexto

O dataset OASIS-Kaggle no esquema de 3 classes (ADR-0001) tem desbalanceamento severo:

| Classe | Pacientes | Slices em train | % no train |
|---|---:|---:|---:|
| non_demented | 185 | 46.726 | 77,5% |
| very_mild | 40 | 9.516 | 15,8% |
| mild_or_moderate | 17 | 4.087 | 6,8% |

Razão entre maior e menor classe em slices: ~11,4×. Sem mitigação, o modelo é incentivado a polarizar em `non_demented` — atinge ~77% de acurácia bruta apenas predizendo a classe majoritária.

Os papers selecionados pelo aluno em `docs/Papers/Trabalhos Relacionados/` foram auditados (F-0009). Conclusões relevantes:
- **Não há consenso explícito** em ViT+Alzheimer sobre `weighted loss` vs `focal loss` vs `sampler balanceado`. A maioria mitiga via `split rigoroso + métrica balanceada + augmentation` (Leveraging Swin é o exemplo mais sólido).
- O paper "estado da arte" no nosso exato dataset (RanCom-ViT, F-0008) **simplesmente ignora** o desbalanceamento e reporta acurácia inflada (99,54%) por causa de split inválido.
- `focal loss` (Lin et al. 2017) aparece em segmentação, não classificação multi-classe Alzheimer.

## Alternativas consideradas

### A — Sem mitigação (CrossEntropyLoss puro)
- **Prós:** simples, replica o RanCom-ViT (controle).
- **Contras:** modelo pode colapsar para classe majoritária, degradando macro-F1 e balanced accuracy. Defendê-lo na banca exigiria justificar por que 77% de acurácia trivial é base aceitável.

### B — Weighted CrossEntropyLoss com pesos `balanced`
- Pesos calculados via fórmula `sklearn.utils.class_weight.compute_class_weight('balanced')` = `n_total / (n_classes × n_class)`.
- **Prós:** padrão de referência fora do nicho ViT+Alzheimer; matematicamente derivado (não-paramétrico); penaliza erros nas minoritárias proporcionalmente à raridade; integra nativamente com `nn.CrossEntropyLoss(weight=...)`.
- **Contras:** pode ampliar instabilidade no início do treino se gradiente for muito sensível à minoritária (mitigável com warmup / lr menor).

### C — Focal Loss (Lin et al. 2017)
- Fator `(1 - p_t)^γ` reduz contribuição de exemplos fáceis e foca nos difíceis.
- **Prós:** popular em detecção de objetos.
- **Contras:** introduz hiperparâmetros (`γ`, `α`) sem heurística clara para classificação Alzheimer 2D MRI; **não há precedente nos papers da pasta** (apenas em segmentação celular do survey Shamshad). Defendê-la exigiria comparar com weighted CE — overhead.

### D — WeightedRandomSampler
- Balanceia a composição dos batches via reposição.
- **Prós:** alternativa popular à weighted loss.
- **Contras:** a unidade de variação verdadeira é o **paciente**, não o slice. Com 17 pacientes em `mild_or_moderate`, o sampler vai re-amostrar os mesmos 17 pacientes várias vezes — não cria diversidade real (apenas vê os mesmos pacientes mais epochs). Risco de overfit nessa classe minoritária.

### E — Oversampling explícito (duplicar minoritárias no dataset)
- **Prós:** trivial.
- **Contras:** mesmo problema da opção D em escala maior; e duplica fisicamente os arquivos no manifesto, complicando reprodutibilidade.

### F — Combinação (sampler balanceado + weighted CE)
- **Contras:** dobra o efeito de penalização sobre minoritárias, podendo causar oscilação no treino. Difícil isolar contribuição de cada técnica em ablações.

## Decisão

Optamos por **B — Weighted CrossEntropyLoss "balanced"** como configuração principal, com **A (sem peso) registrado como ablação obrigatória**.

Implementação:

1. Pesos calculados **uma única vez** a partir do split de treino (não global ao manifesto), via fórmula `sklearn`:
   `w_c = n_train / (num_classes × n_train_in_class_c)`
2. Pesos passados para `nn.CrossEntropyLoss(weight=tensor_w)`.
3. Pesos salvos em `experiments/splits/split_v1.json` ou em `experiments/<run>/config.yaml` para auditoria.
4. **Ablação obrigatória:** mesmo modelo, mesmo split, mesmas seeds, sem peso. Reportar lado a lado para mostrar quanto a métrica balanceada melhora.
5. **Não usar focal loss, sampler balanceado nem oversampling agora.** Ficam como trabalho futuro/ablação opcional caso a banca peça.

### Cálculo concreto para o split v1

| Classe | n_train_slices | peso = 60.329 / (3 × n) |
|---|---:|---:|
| non_demented | 46.726 | 0,4304 |
| very_mild | 9.516 | 2,1135 |
| mild_or_moderate | 4.087 | 4,9210 |

Erros em `mild_or_moderate` contribuem ~11,4× mais para a loss do que erros em `non_demented` — espelhando o desbalanceamento exato.

## Consequências

- **Treinos rodam em pares** (com peso vs sem peso), aumentando custo computacional do baseline em ~2×. Cabe no Colab T4 dado que o baseline ResNet-50 é leve (~25M params).
- **Métricas reportadas** (ADR-0005): macro-F1, balanced accuracy, AUC macro. Acurácia bruta também aparece para comparabilidade com RanCom-ViT, mas explicitamente marcada como secundária.
- **Risco de instabilidade no início do treino:** mitigado por LR pequeno (~1e-4 para fine-tuning timm), warmup de 1 epoch, e clipping de gradiente padrão (`max_norm=1.0`). A serem fixados em ADR-0008 (treino) quando implementado.
- **Decisão revisitável** se na ablação o ganho de B sobre A for marginal ou negativo: documentar com gráficos de loss/balanced accuracy por epoch, e considerar sampler balanceado como alternativa.
- **Não usar weighted loss em test set** — métricas de avaliação são sobre as previsões puras, não sobre a loss ponderada. (Risco de auditoria caso alguém olhe o código de test.)

## Referências

- Findings: F-0008 (RanCom-ViT como caso negativo), F-0009 (varredura na literatura).
- ADR-0001 (3 classes), ADR-0002 (split por paciente), ADR-0005 (métricas), ADR-0006 (augmentation).
- `sklearn.utils.class_weight.compute_class_weight('balanced')`: fórmula derivada de King & Zeng (2001).
- Lin et al., 2017 — *Focal Loss for Dense Object Detection* — discutido como alternativa C, descartado.
- Buda et al., 2018 — *A systematic study of the class imbalance problem in convolutional neural networks* — recomenda oversampling para CNN, mas em datasets onde a unidade real de variação é o paciente, conclusões precisam ser adaptadas.
