# F-0020 — Análise qualitativa dos mapas de interpretabilidade (Swin-T V3, Grad-CAM)

- **Data:** 2026-05-23
- **Status:** Confirmed (inspeção visual dos overlays do modelo campeão)
- **Categoria:** clinico

## O que descobrimos

Os mapas Grad-CAM do Swin-T V3 (modelo campeão, F-0019) mostram **evidência clínica positiva no caso de demência clara**: o modelo foca nos ventrículos laterais / região periventricular — marcador real de atrofia em Alzheimer. Para as classes sutis (very_mild, non_demented) os mapas são difusos. Há também uma limitação técnica de resolução (Grad-CAM 7×7 do Swin).

## Evidência

Mapas gerados localmente via `src/interpretability/generate_maps.py` sobre o checkpoint real `swin_tiny_v3_class3_weighted_kaggle`, 4 amostras por classe do test. Arquivos em `results/interpretability/<run>/` (versionados).

| Amostra | Classe real | Onde o Grad-CAM ativa | Plausibilidade clínica |
|---|---|---|---|
| OAS1_0316 | mild_or_moderate | **ventrículos laterais centrais + periventricular** (vermelho focado) | ✅ alta — dilatação ventricular por atrofia é marcador de AD |
| OAS1_0243 | very_mild | centro frio; ativação difusa em bordas/fundo | ⚠️ baixa |
| OAS1_0156 | non_demented | difuso + padrão de blocos no fundo | ⚠️ baixa (parcial artefato) |

## Interpretação

1. **Sinal clínico real no caso severo.** Em `mild_or_moderate`, o modelo atende exatamente as estruturas que a literatura associa a Alzheimer (ventrículos dilatados, região periventricular). É evidência de que o Swin-T V3 aprendeu representação clinicamente significativa — não atalho. **Figura-estrela da seção de interpretabilidade do TCC.**
2. **Mapas difusos nas classes sutis.** very_mild e non_demented produzem mapas pouco focados. Consistente com (a) a dificuldade clínica intrínseca do estágio inicial (atrofia sutil) e (b) as métricas mais baixas dessas classes (F1 very_mild 0,384; F-0019). O modelo "não sabe bem onde olhar" justamente onde a doença é sutil — coerente.
3. **Limitação técnica de resolução.** Grad-CAM do Swin opera no último estágio (grade 7×7), upsampled para 224×224 → mapas grosseiros + padrão de blocos no fundo preto (área pós skull-strip). É artefato do upsample, não sinal. Mencionar como limitação metodológica.

## Implicação

- **Para o TCC:** usar o overlay de `mild_or_moderate` (OAS1_0316) como evidência principal de interpretabilidade clínica; discutir honestamente a difusão nas classes sutis e a limitação de resolução do Grad-CAM do Swin.
- **Próximo passo recomendado:** gerar também (a) **Attention Rollout do ViT-Base** (grade 14×14 → mapas mais finos; método nativo de transformer) e (b) **Grad-CAM do ResNet-50**, para comparação "onde cada arquitetura olha". O Rollout do ViT pode dar mapas mais limpos que o Grad-CAM 7×7 do Swin.
- **Melhoria opcional (não-prioritária):** mascarar o fundo (área = 0 pós skull-strip) no overlay para remover o padrão de blocos — legítimo (fundo não tem informação), mas aplicar com nota metodológica para não parecer "maquiagem".

## Notas / armadilhas

- **Não generalizar de poucas amostras.** Esta análise olhou 3-4 casos por classe. Para a escrita final, inspecionar mais amostras e selecionar exemplos representativos (incluindo erros) com critério, não cherry-picking do melhor.
- **Mapa cru é o honesto.** Mostrar o Grad-CAM sem mascarar o fundo na análise; se mascarar para a figura final, declarar.
- **Atenção (lição F-0003):** a leitura "ventrículos" foi feita olhando o corte axial real (estrutura central em asa de borboleta = ventrículos laterais). Confirmar a anatomia com o orientador antes de afirmar categoricamente no texto.
- Checkpoints `.pt` não versionados (gitignore); as figuras `.png` de interpretabilidade sim.
