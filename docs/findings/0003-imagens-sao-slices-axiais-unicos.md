# F-0003 — Imagens 496×248 são slices axiais únicos (não dual-view)

- **Data:** 2026-05-10
- **Status:** Confirmed (após erro inicial, ver "Notas")
- **Categoria:** imagem

## O que descobrimos

Cada arquivo `.jpg` em `Data/` é **um único slice axial** alongado horizontalmente. Dimensões e propriedades:

- **Tamanho:** 496×248 pixels (largura × altura), aspect ratio exato 2:1 em 100% das amostras checadas.
- **Modo:** RGB com **R==G==B** em 100% das amostras (grayscale promovido pela conversão NIfTI→JPG; 3 canais idênticos, 1 canal real de informação).
- **Aspecto alongado vem do resampling NIfTI→JPG do uploader Kaggle** — provavelmente reflete voxels não-isotrópicos do MRI MPRAGE original. Não é distorção patológica do dataset; é apenas como ele foi exportado.
- **Conteúdo:** vista axial mostrando cérebro inteiro. Em fatias baixas (índices ~100-110) os globos oculares aparecem proeminentemente na metade direita anatômica do paciente — isso é o que **incorretamente** levou a uma hipótese inicial de "dual-view" (axial + sagital concatenados).

## Evidência

- Script: `src/data/inspect_image_layout.py` checa R==G==B em 8 amostras (2 por classe), salva grade visual em `results/eda/figures/sample_grid.png`.
- 8/8 imagens com R==G==B confirmado.
- Aspect ratio médio: 2.000 (exato).
- Confirmação visual via amostra enviada pelo aluno mostrando claramente um único corte axial.

## Implicação

- **Pré-processamento simplificado:** resize direto para 224×224 (squash) e normalização ImageNet. Sem necessidade de cropping ou separação de vistas. Ver ADR-0004.
- Para conversão a 1 canal: viável tecnicamente (todos os 3 canais são idênticos), **mas** quebraria compatibilidade com pesos pré-treinados ImageNet → mantemos os 3 canais.
- Augmentations seguras anatomicamente: flip horizontal, rotação leve (±5°), jitter de brilho/contraste. **Vertical flip é incorreto** (cabeça invertida não ocorre clinicamente).

## Notas / armadilhas

- **Erro grave que cometi nesta auditoria:** ao ver 496×248, presumi "duas vistas concatenadas" e construí uma análise quantitativa elaborada (MAE par-a-par entre metades) sustentando essa hipótese. Os "perfis sagitais com nariz/queixo" eram, na verdade, globos oculares numa fatia axial baixa. O `diff_left_right ~54` era simplesmente porque hemisférios esquerdo e direito do cérebro têm anatomia diferente — trivial num axial.
- **Lição registrada permanentemente** em memória de feedback (`feedback_tcc_rigor.md`): quando aspect ratio for inesperado, **abrir a imagem visualmente antes de teorizar**. Hipóteses exóticas devem ser comparadas com a interpretação mais simples antes de virarem decisão de pipeline.
- Esse erro foi pego pelo aluno em ~30 minutos. Em ambiente de pesquisa real teria contaminado o pipeline por dias e potencialmente ido para a banca. Reforça a importância de revisão humana mesmo em decisões "puramente técnicas".
