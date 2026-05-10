# F-0009 — Como a literatura de ViT em imagem médica trata desbalanceamento de classes

- **Data:** 2026-05-10
- **Status:** Confirmed (varredura sistemática nos 14 PDFs em `docs/Papers/Trabalhos Relacionados/`)
- **Categoria:** trabalho-relacionado

## O que descobrimos

Varredura automática (`src/data/survey_imbalance_handling.py`) por termos relacionados a tratamento de desbalanceamento (`class imbalance`, `weighted loss`, `focal loss`, `balanced accuracy`, `oversampl`, `WeightedRandomSampler`, etc.) nos 14 papers da pasta. Excertos completos em `_imbalance_survey_excerpts.md`. Síntese:

### Padrões observados

| Paper | Estratégia para desbalanceamento | Métricas | Split |
|---|---|---|---|
| **RanCom-ViT** (Lu 2025, *mesmo dataset que nós*) | **Nenhuma** — só reconhece o desbalanceamento e mantém | Acc, Prec, Sens, Spec, F1 simples | Aleatório por classe (F-0008) |
| **Leveraging Swin** (multi-shell dMRI ADNI) | Nenhuma loss-weighting; **mitiga via split rigoroso + balanced accuracy** | **Balanced accuracy** primária + F1 + recall | Subject-wise hold-out 15% + 5-fold CV agrupado por sujeito |
| **Vision_transformer_equipped_CNN** | Augmentations agressivas (flip, affine, elastic, blur, motion, ghosting, noise) | Não detalhado no excerto | Não detalhado |
| **Joint_transformer (3D)** | (sem ocorrências dos termos) | — | — |
| **VGG-TSwinformer** | (sem ocorrências dos termos) | — | — |
| **Survey "Transformers in Medical"** (Shamshad 2023) | Menciona `focal loss` em segmentação celular (Cell-DETR), `weighted loss` em uncertainty-guided report generation | Cobertura ampla, sem recipe específica | — |
| **ResNet** (He 2016) | Standard data augmentation (flip, color, scale crop) | Top-1/Top-5 accuracy | ImageNet (balanceado) |
| **ViT/Swin originais** | Standard ImageNet augmentations | Top-1 accuracy | ImageNet (balanceado) |

### Conclusões da varredura

1. **Em ViT aplicado a Alzheimer 2D MRI, weighted loss explicit não é o padrão.** A combinação predominante é: `split rigoroso` + `balanced accuracy / macro-F1` + `data augmentation`. Class weights aparecem mais em segmentação (foreground/background imbalance é estrutural lá).
2. **Balanced accuracy é a métrica mais usada** em trabalhos rigorosos como Leveraging Swin (múltiplas tabelas reportam balanced accuracy como primária).
3. **Augmentations padrão são leves** (flip, rotação, jitter); o paper Vision_transformer_equipped propõe pacote mais agressivo (elastic deformation, blur, ghosting). Para nossa imagem MRI alongada 496×248 com aspect ratio fixo após resize, augmentations geométricas pesadas (elastic) podem distorcer estruturas-alvo (hipocampo) — manter pacote leve.
4. **Focal loss aparece em segmentação**, não em classificação multi-classe Alzheimer. Adotá-la introduziria escolhas de hiperparâmetro (γ, α) a defender sem ganho consolidado em literatura específica.
5. **Oversampling de slices não cria diversidade real** quando o número de pacientes minoritários é muito pequeno (no nosso caso, 23 pacientes em `mild_or_moderate`). Repetir os mesmos 23 pacientes não é regularização — é só ver os mesmos 23 mais vezes.

## Evidência

- Varredura: `src/data/survey_imbalance_handling.py`.
- Excertos consolidados: `docs/findings/_imbalance_survey_excerpts.md` (14 papers, 65 excertos auditados).
- Termos buscados: ver lista no script.
- Para Leveraging Swin (referência metodológica positiva), trecho-chave (página 3 do PDF):
  > "15% of unique subjects were randomly assigned to a fixed hold-out test set […]. The remaining 85% of subjects were used for 5-fold grouped and stratified cross-validation. This strategy ensured that all data from any given subject appeared entirely within either the training or validation set for a fold, eliminating optimistic bias from intra-subject correlations."

## Implicação

A decisão para o nosso TCC fica clara e defensável (formalizada em ADR-0007):

1. **Métrica primária: balanced accuracy + macro-F1 + AUC** (alinhado a Leveraging Swin; já fixado em ADR-0005).
2. **Loss principal: weighted CrossEntropyLoss com pesos 'balanced'** da fórmula sklearn (`weight_i = N_total / (n_classes × n_class_i)`). É a opção mais simples, derivada matematicamente, não-paramétrica (sem `γ` para sintonizar). Cobre o caso onde a literatura **Alzheimer/ViT** não deu recipe explícita mas o nosso desbalanceamento é severo (~13:1 em slices).
3. **Ablação: re-treinar sem peso**, para fundamentar empiricamente o ganho do peso (vs paper RanCom-ViT que ignora o problema completamente).
4. **Não usar focal loss, oversampling de slices, ou WeightedRandomSampler.** Justificativas registradas no ADR-0007.
5. **Augmentations leves já fixadas em ADR-0006** (flip horizontal, rotação ±5°, jitter brilho/contraste) cobrem regularização adicional sem distorcer anatomia.

## Notas / armadilhas

- **Limitação da varredura:** termos podem aparecer com hifenização ou separação por quebra de página, escapando do regex. A varredura serve como triagem; não substitui leitura completa de papers que são especialmente alinhados ao nosso problema. Os papers principais (RanCom-ViT, Leveraging Swin) foram lidos com mais profundidade.
- **Possível viés de seleção:** os 14 papers em `docs/Papers/Trabalhos Relacionados/` foram escolhidos pelo aluno, predominantemente sobre ViT/Swin/interpretabilidade em Alzheimer. Não cobre toda a literatura de classificação médica desbalanceada (ex: Lin et al. 2017 Focal Loss, ou trabalhos clássicos sobre SMOTE), mas representa bem o nicho ViT+Alzheimer 2D.
- **Atenção em revisões posteriores:** se algum paper que cite o RanCom-ViT propuser metodologia rigorosa nos mesmos dados, ele se torna referência ainda mais relevante. Vale uma busca semântica posterior por "OASIS Kaggle ninadaithal vision transformer subject-level" no Google Scholar.
