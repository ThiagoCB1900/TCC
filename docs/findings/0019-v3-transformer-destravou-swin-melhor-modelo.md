# F-0019 — V3-transformer destravou ganho; Swin-T V3 é o melhor modelo do projeto

- **Data:** 2026-05-23
- **Status:** Confirmed (ViT-Base V3 e Swin-T V3 weighted no Kaggle T4, mesmo split_v1)
- **Categoria:** metodologia

## O que descobrimos

A regularização forte do V3-transformer (ADR-0013: lr_backbone 5e-6, drop_path 0.2, weight_decay 0.2, warmup 3) **atenuou o overfit e melhorou ambos os transformers**, confirmando a hipótese de F-0018 (havia ganho preso atrás do overfit). **Swin-T V3 é o melhor modelo do projeto** em todas as métricas (test balanced_acc 0,616, macro_f1 0,551, AUC 0,836), significativamente superior ao ViT V3 (McNemar p≈0).

## Evidência

### Resultados consolidados (test, n=13.237, mesmo split_v1, weighted CE)

| Modelo | accuracy | balanced_acc | macro_f1 | auc_macro | best_ep | F1 mild+mod |
|---|---:|---:|---:|---:|---:|---:|
| ResNet-50 V2 | 0,681 | 0,587 | 0,513 | 0,804 | 11 | 0,323 |
| ViT-Base V2 | 0,730 | 0,521 | 0,483 | 0,822 | 2 | 0,293 |
| ViT-Base V3 | 0,692 | 0,568 | 0,514 | 0,811 | 2 | 0,354 |
| Swin-T V2 | 0,723 | 0,594 | 0,528 | 0,823 | 2 | 0,374 |
| **Swin-T V3** | 0,720 | **0,616** | **0,551** | **0,836** | 4 | **0,426** |

### Curvas — overfit atenuado

- **Swin-T V3:** best epoch 4 (era 2 no V2); val_loss pico ~1,5 (era 3,0); bal_acc val sobe até 0,694 (ep4). Treino mais saudável.
- **ViT-Base V3:** best epoch ainda 2, mas val_loss pico ~2,1 (era 3,0); train_loss cai mais devagar. Overfit atenuado, não eliminado — ViT-Base (86M) segue o mais difícil.

### McNemar (predições por amostra, mesmo test)

| Comparação | b | c | p | Veredito |
|---|---:|---:|---:|---|
| Swin V3 vs ViT V3 | 445 | 809 | ≈0 | **Swin V3 significativamente melhor** |
| ViT V3 vs ViT V2 | 1129 | 620 | ≈0 | **V3 melhorou o ViT significativamente** |
| Swin V3 vs Swin V2 | 390 | 343 | 0,089 | melhora real, mas **não significativa** |

## Interpretação

1. **Swin-T V3 é o modelo campeão** (bal_acc 0,616, AUC 0,836, F1 minoritária 0,426). Supera o baseline ResNet (0,587) por +0,029 e o ViT V3 com significância estatística.
2. **A regularização V3 ajudou mais o ViT** (+0,047, significativo) que o Swin (+0,022, n.s.) — coerente: o ViT overfittava mais (era o que mais tinha a ganhar com regularização).
3. **Swin > ViT consistente e significativo.** Swin hierárquico (inductive bias de localidade, janelas deslizantes) generaliza melhor que ViT puro neste regime de dados pequenos (242 pacientes). É um achado interpretável e defensável: ViT puro é mais "faminto por dados".
4. **ViT-Base segue subaproveitado** (best epoch 2 mesmo com V3) — 86M params é muito para o dataset. Conclusão honesta do TCC, não falha de implementação.
5. **Teto subiu de ~0,59 (CNN) para ~0,62 (Swin V3); val pico 0,694.** A regularização elevou o teto. Para subir mais, a alavanca restante é **dados** (2.5D, ADR-0011) — regularização e arquitetura já foram exploradas.
6. **AUC do Swin V3 = 0,836** (melhor do projeto) — o modelo discrimina bem as classes; o limite restante está na separabilidade intrínseca de fatias 2D únicas.

## Implicação

- **Swin-T V3 é o modelo a levar adiante** para interpretabilidade (Attention Rollout — pilar inegociável do TCC) e como candidato para 2.5D.
- **Comparação honesta para o TCC** está pronta: ResNet vs ViT vs Swin, todos sob a mesma metodologia rigorosa (split por paciente, weighted CE, métricas balanceadas), com McNemar. Contraste com RanCom-ViT (99,54% acc por split de slice) é a tese central.
- **Decisão de próximo passo** (cronograma: ~11 dias até entrega da cópia, 2026-06-03): priorizar **interpretabilidade** (regra crítica nº 3, pilar inegociável) sobre 2.5D (melhoria de métrica de ganho incerto). 2.5D fica reservado para depois da cópia se houver tempo.
- **Pendência para McNemar completo:** ResNet V2 não tem predições salvas (treinado antes da persistência). Re-rodar ResNet V2 (~1h) com o código atual para incluí-lo no McNemar das 3 arquiteturas — fazer antes da escrita final dos resultados.

## Notas / armadilhas

- **Swin V3 vs V2 é n.s. (p=0,089)** — ao escrever, não afirmar que "V3 é significativamente melhor que V2 no Swin". Afirmar: "V3 deu o melhor resultado e treino mais estável (best epoch 4 vs 2); a diferença para o V2 está dentro da margem estatística". Honestidade.
- **Não comparar nossos números com a literatura ADNI/dMRI** (Vision-ViT-CNN 92%, Leveraging Swin 95%) como se fosse o mesmo problema — datasets/modalidades/tarefas diferentes (F-0015). Comparar contra nós mesmos + contraste metodológico com RanCom-ViT.
- Tempos V3: ViT ~? (best ep2, parou cedo), Swin rodou 9 epochs. Dentro da cota.
