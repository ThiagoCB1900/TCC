# F-0022 — Ablação de capacidade: ResNet-18 (menos params) é PIOR; hipótese "menos parâmetros generaliza melhor" refutada

- **Data:** 2026-05-24
- **Status:** Confirmed (ResNet-18 weighted no Kaggle T4, mesmo split_v1, mesma metodologia V2)
- **Categoria:** metodologia

## O que descobrimos

A hipótese de que "a ResNet-50 overfittaria por ter muitos parâmetros, e uma rede menor generalizaria melhor" foi **refutada empiricamente**. A ResNet-18 (~11,2M params, ~metade da ResNet-50) — com transfer learning e a mesma metodologia V2 — ficou a **pior de todas as arquiteturas** (test balanced_acc 0,501), significativamente abaixo da ResNet-50 (0,587; McNemar p=2,8e-6). Confirma que o overfit **não era** o gargalo da ResNet-50 e que o limite é a representação (dados), não a capacidade.

## Evidência

### Ranking final (test, split_v1, weighted CE, metodologia V2)

| Modelo | params | accuracy | balanced_acc | macro_f1 | AUC | F1 mild+mod | best_ep |
|---|---:|---:|---:|---:|---:|---:|---:|
| Swin-T V3 | 27,5M | 0,720 | **0,616** | 0,551 | 0,836 | 0,426 | 4 |
| ResNet-50 V2 | 23,5M | 0,681 | 0,587 | 0,513 | 0,804 | 0,323 | 11 |
| ViT-Base V3 | 85,8M | 0,692 | 0,568 | 0,514 | 0,811 | 0,354 | 2 |
| **ResNet-18** | 11,2M | 0,666 | **0,501** | 0,458 | 0,777 | 0,208 | 9 |

Run: `20260524_155530_resnet18_class3_weighted_kaggle`.

### McNemar (ResNet-18 vs demais, mesmo test n=13.237)

| Comparação | b | c | p | Veredito |
|---|---:|---:|---:|---|
| ResNet-18 vs ResNet-50 | 1041 | 837 | 2,8e-6 | ResNet-50 melhor (significativo) |
| ResNet-18 vs ViT-Base | 1045 | 694 | ≈0 | ViT melhor (significativo) |
| ResNet-18 vs Swin-T | 1281 | 566 | ≈0 | Swin melhor (significativo) |

ResNet-18 perde para **todas** as outras, inclusive em accuracy bruta (0,666, a menor).

### Curva (treino saudável, sem overfit)

train_loss 1,08 → 0,38 gradual em 14 epochs; val_loss estável (~0,55-0,61); best epoch 9. **Não houve overfit descontrolado** — a ResNet-18 simplesmente tem teto mais baixo.

## Interpretação

1. **Hipótese refutada com rigor:** menos parâmetros → desempenho **pior**, não melhor. A ResNet-18 é a última colocada em todas as métricas e perde no McNemar para todas com significância.
2. **Confirma o diagnóstico de F-0017:** o gargalo da ResNet-50 **não era overfit/capacidade** — senão reduzir params teria ajudado. O gargalo é a **representação** (fatia axial 2D única). Reduzir capacidade só piora.
3. **Capacidade insuficiente prejudica as classes difíceis:** F1 da minoritária caiu de 0,323 (ResNet-50) para 0,208 (ResNet-18). Com menos capacidade, o modelo "desiste" mais da classe rara.
4. **Nuance val→test:** ResNet-18 teve val pico 0,648 (alto) mas test 0,501 (queda ~0,15, bem maior que ~0,05 da ResNet-50). Sugere generalização mais frágil para pacientes não vistos — mas reportar com cautela (val tem só 52 pacientes; parte pode ser variância). Não superinterpretar.

## Implicação

- **Fecha a análise de arquiteturas/capacidade.** Exploramos: regularização (V1→V2), arquitetura (CNN vs ViT vs Swin), e agora capacidade (ResNet-18 vs 50). Todas convergem: o teto (~0,62 com Swin) é imposto pelos **dados**, não por modelo/hiperparâmetro. A única alavanca restante é **2.5D** (ADR-0011).
- **Discussão do TCC enriquecida:** podemos afirmar, com 4 arquiteturas e McNemar, que (a) o viés local (CNN/Swin) ajuda em dados pequenos; (b) capacidade excessiva (ViT-Base) ou insuficiente (ResNet-18) prejudicam; (c) Swin-T (capacidade média + viés local) é o ponto ótimo no nosso regime.
- **Ranking de arquiteturas consolidado:** Swin-T V3 > ResNet-50 V2 > ViT-Base V3 > ResNet-18 (em balanced_acc), todas com diferenças significativas no McNemar (atenção à leitura accuracy vs balanced — F-0021).

## Notas / armadilhas

- **A intuição do aluno era legítima** (menos params → menos overfit) — só não se aplica aqui porque o overfit já estava controlado e o gargalo é outro. Registrar a refutação é tão valioso quanto uma confirmação.
- Não confundir "ResNet-18 pior" com "transfer learning não importa" — a ResNet-18 USA transfer learning; o que faltou foi capacidade. A ablação de transfer learning (from-scratch) é experimento diferente, ainda não feito.
- ResNet-18 tem `test_predictions.npz` salvo → já entrou no McNemar das 4 arquiteturas.
