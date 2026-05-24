# ADR-0014 — Métodos de interpretabilidade: Attention Rollout (ViT) + Grad-CAM (ResNet/Swin)

- **Data:** 2026-05-23
- **Status:** Accepted
- **Decisores:** Thiago, Claude

## Contexto

Interpretabilidade é pilar inegociável do TCC (regra crítica nº 3) e diferencial frente aos trabalhos que só reportam métricas (incluindo o RanCom-ViT, sem interpretabilidade). Com os modelos treinados (Swin-T V3 campeão, F-0019), precisamos gerar explicações visuais defensáveis na banca, conectadas aos papers que o aluno já tem.

## Decisão

Cada arquitetura usa o método de atribuição mais natural a ela:

| Modelo | Método | Paper-âncora | Camada-alvo |
|---|---|---|---|
| ViT-Base/16 | **Attention Rollout** | Abnar & Zuidema 2020 (`attention_flow_in_transformers.pdf`) | matrizes de atenção de todos os 12 blocos |
| ResNet-50 | **Grad-CAM** | Selvaraju et al. 2017 | `layer4` (última conv, 7×7) |
| Swin-T | **Grad-CAM** (+ reshape) | Selvaraju 2017 | último estágio (`layers[-1]`, reshape de tokens) |

**Por que essa divisão:**
- **Attention Rollout** é o método nativo de ViT: agrega a atenção entre camadas (com conexão residual via identidade) para estimar a contribuição de cada patch ao token CLS. É o que conecta diretamente com a literatura de interpretabilidade de transformers do projeto.
- **Grad-CAM** é o padrão para CNN (ResNet). Para Swin, a atenção é por janelas deslizantes — Attention Rollout puro é mal-definido; Grad-CAM sobre o último estágio (com reshape de tokens para grade) é mais robusto e é prática consolidada (`pytorch-grad-cam` usa a mesma abordagem).
- **Chefer et al. 2021** (CVPR, `Chefer_...pdf`) é uma alternativa mais sofisticada (relevância por LRP+gradiente) reservada como melhoria se o Attention Rollout puro for ruidoso.

### Implementação (sem dependências novas)

- `src/interpretability/attention_rollout.py` — captura a matriz de atenção via forward-hook no `attn_drop` de cada bloco (com `fused_attn=False`); rollout com residual; mapa do CLS → grid 14×14.
- `src/interpretability/gradcam.py` — Grad-CAM genérico com `reshape_transform` opcional (para Swin tokens → grade).
- `src/interpretability/generate_maps.py` — CLI que carrega `checkpoint_best.pt`, reconstrói o modelo via factory, seleciona N amostras por classe do test e salva overlays + grid.
- Overlay via OpenCV (colormap JET, alpha 0.5) — já no `requirements.txt`.

### Onde roda

**Local, em CPU** — interpretabilidade é leve (forward de poucas imagens + hooks). Fluxo: baixar `checkpoint_best.pt` da run do Kaggle para `experiments/runs/<run>/`, rodar `python -m src.interpretability.generate_maps --run-dir ...`. Não precisa de GPU/Kaggle.

## Consequências

- Os 3 pilares do TCC ficam entregues: baseline ResNet (✓), ViT/Swin (✓), interpretabilidade (✓).
- Mapas gerados para o **Swin-T V3** (campeão) são o material principal; ResNet (Grad-CAM) e ViT (Rollout) entram como comparação de "onde cada arquitetura olha".
- Análise qualitativa esperada: verificar se os modelos atendem a regiões clinicamente plausíveis (hipocampo, ventrículos, córtex temporal medial) — e mostrar exemplos de acertos e erros.
- Figuras de interpretabilidade **são versionadas** (`results/interpretability/`, exceção no `.gitignore`) — poucas e centrais para o TCC.
- Validação obrigatória (lição F-0003): abrir e inspecionar os mapas antes de afirmar conclusões clínicas. Com checkpoint dummy (pretrained) os mapas não são significativos — só com o checkpoint fine-tuned a leitura clínica é válida.

## Referências

- Abnar & Zuidema 2020 — Attention Rollout/Flow (`docs/Papers/.../attention_flow_in_transformers.pdf`)
- Selvaraju et al. 2017 — Grad-CAM (`docs/Papers/.../Selvaraju_Grad-CAM_...pdf`)
- Chefer et al. 2021 — Transformer interpretability beyond attention (alternativa, `docs/Papers/.../Chefer_...pdf`)
- F-0019 (Swin-T V3 campeão), regra crítica nº 3 do CLAUDE.md.
