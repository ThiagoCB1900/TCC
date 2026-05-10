# ADR-0004 — Resize direto para 224×224, manter RGB sintético

- **Data:** 2026-05-10
- **Status:** Accepted
- **Decisores:** Thiago, Claude

## Contexto

As imagens do dataset Kaggle são **496×248 RGB** (aspect ratio 2:1) com R==G==B (grayscale promovido pela conversão NIfTI→JPG). Cada arquivo é **um único slice axial alongado horizontalmente** — não há duas vistas concatenadas como inicialmente suspeitamos (ver F-0003 para o histórico desse erro).

Modelos pré-treinados ImageNet (ViT-Base/16, Swin-T, ResNet-50) esperam tipicamente entrada 224×224 RGB com normalização ImageNet (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).

## Alternativas consideradas

- **A — Resize squash 224×224 (deformação).** Distorce o cérebro horizontalmente para encolhê-lo, mas mantém todo o conteúdo. Padrão da literatura em ViT médico.
- **B — Resize com letterbox (padding).** Mantém aspect ratio adicionando barras pretas. Reduz resolução efetiva do cérebro (de ~248px de altura para ~112px após escala que preserva 2:1).
- **C — Resize 384×384** (suportado por ViT-Base e Swin com pesos próprios). Mais resolução, mais memória, mais tempo. Justificado se 224×224 perder informação relevante.
- **D — Converter para 1 canal (grayscale puro).** Faria sentido conceitualmente já que R==G==B, mas pesos pré-treinados ImageNet são para 3 canais. Adaptar seria perder transfer learning.

## Decisão

Optamos por **A — resize squash para 224×224** como configuração padrão, **mantendo os 3 canais RGB sintéticos** para compatibilidade com pesos ImageNet:

- `Resize((224, 224))` direto. Modelos aprendem invariância à deformação horizontal já que o squash é consistente entre todas as imagens.
- Normalização ImageNet (`mean/std` padrão) para manter compatibilidade com pesos pré-treinados.
- Durante treino, augmentations leves (flip horizontal, pequena rotação ±5°, jitter de brilho/contraste) — todas aplicadas **dentro do dataloader, só no split de treino** (ver ADR-0002).
- 384×384 fica como ablação opcional para Swin-T se a performance em 224 ficar abaixo do esperado.

## Consequências

- Pipeline simples e alinhado com 90% dos papers de ViT em domínio médico.
- Se a deformação horizontal prejudicar regiões específicas (ex: hipocampo simétrico), os mapas de atenção podem indicar — vai ser detectável na fase de interpretabilidade (semana 4).
- Decisão revisitável: se 224 squash decepcionar, ADR-0006 (a criar) pode trocar para letterbox 224 ou 384 isotrópico.
- Augmentation horizontal flip é seguro: cérebro tem aproximada simetria sagital. Augmentation vertical flip seria errado anatomicamente — **não usar**.

## Referências

- Findings: F-0003 (estrutura real das imagens)
- ViT (Dosovitskiy 2020) — entrada 224×224 padrão
- Swin (Liu 2021) — entrada 224×224 ou 384×384 padrão
- timm docs: https://github.com/huggingface/pytorch-image-models
