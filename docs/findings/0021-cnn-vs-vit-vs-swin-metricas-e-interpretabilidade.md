# F-0021 — CNN vs ViT vs Swin: métricas e interpretabilidade convergem (ViT data-hungry)

- **Data:** 2026-05-23
- **Status:** Confirmed (mapas dos 3 modelos no mesmo paciente + métricas no mesmo test)
- **Categoria:** metodologia

## O que descobrimos

As três linhas de evidência — **métricas, interpretabilidade e teoria** — convergem para a mesma conclusão: no nosso regime de dados pequenos (242 pacientes train), a **CNN (ResNet-50) supera o ViT puro**, e o **Swin** (transformer com viés local) é o melhor. O ViT puro nem performa melhor nem aprende a olhar as estruturas clinicamente relevantes. Isso confirma a natureza *data-hungry* do ViT descrita por Dosovitskiy et al. (2020).

## Evidência

### 1. Métricas (test, split_v1, weighted CE)

| Modelo | tipo | **accuracy** | **balanced_acc** | macro_f1 | AUC | onde olha |
|---|---|---:|---:|---:|---:|---|
| ResNet-50 V2 | **CNN pura** | 0,681 | **0,587** | 0,513 | 0,804 | ventrículos (foco limpo) |
| ViT-Base/16 V3 | **Transformer puro** | 0,692 | 0,568 | 0,514 | 0,811 | atenção difusa, bordas/cantos |
| Swin-T V3 | **Transformer hierárquico** | **0,720** | **0,616** | **0,551** | **0,836** | ventrículos (+ ruído de blocos) |

**Inversão crítica accuracy ↔ balanced_acc (ResNet vs ViT):** o ViT tem **accuracy bruta maior** (0,692 > 0,681) mas **balanced_acc menor** (0,568 < 0,587) que a CNN. O ViT acerta mais no total **porque polariza para a classe majoritária** (non_demented, 79% do test); quando se dá peso igual a cada classe (balanced_acc), a CNN vence. Isso é a essência do problema de desbalanceamento (ADR-0005): accuracy bruta engana.

### 1b. McNemar (predições por amostra, mesmo test, n=13.237)

| Comparação | b | c | p | Veredito (sobre **accuracy bruta**) |
|---|---:|---:|---:|---|
| ResNet vs ViT | 1025 | 878 | 0,0008 | ViT acerta mais no total (significativo) — mas via classe majoritária |
| ResNet vs Swin | 1271 | 760 | ≈0 | Swin significativamente melhor |
| ViT vs Swin | 809 | 445 | ≈0 | Swin significativamente melhor |

**Atenção metodológica:** McNemar testa **acertos brutos**, então favorece quem acerta mais a classe majoritária. Por isso "ResNet vs ViT" dá ViT — o oposto do ranking por balanced_acc. **Reportar McNemar SEMPRE ao lado da balanced_acc**, nunca isolado, sob risco de conclusão invertida em dados desbalanceados. Swin domina em ambas as leituras (accuracy E balanced) — sua superioridade é inequívoca.

### 2. Interpretabilidade — mesmo paciente (OAS1_0316, mild_or_moderate)

Inspeção visual dos overlays em `results/interpretability/*/08_mild_or_moderate_OAS1_0316.png`:

- **ResNet-50 (Grad-CAM):** mancha quente concentrada nos **ventrículos laterais / região periventricular**. Foco limpo, fundo frio.
- **Swin-T (Grad-CAM):** também foca os **ventrículos centrais**, com algum padrão de blocos (resolução 7×7).
- **ViT-Base (Attention Rollout):** atenção **espalhada** por todo o cérebro, com focos quentes em **bordas/cantos** (fora do parênquima). **Não localiza** as estruturas centrais.

### 3. Teoria

Dosovitskiy et al. (2020, paper do ViT em `docs/Papers/`): ViT só supera CNNs quando pré-treinado em datasets enormes (JFT-300M); abaixo disso, **a falta de inductive bias (localidade, equivariância a translação) faz o ViT perder para CNNs**. Swin (Liu et al. 2021) reintroduz o viés local via janelas deslizantes → melhor em dados limitados.

## Interpretação

1. **Swin-T V3 é inequivocamente o melhor** — vence em accuracy E balanced_acc, com significância (McNemar p≈0 contra ambos). Conclusão robusta.
2. **CNN vs ViT é nuançado (não "CNN simplesmente vence"):** o ViT tem accuracy bruta maior, mas **balanced_acc menor** — ele troca desempenho nas classes minoritárias por acertos na majoritária (polariza). Em uma tarefa clínica onde detectar demência (minoritárias) importa, **a balanced_acc é a métrica relevante, e nela a CNN supera o ViT**. A frase honesta é: *"o ViT puro tende a polarizar para a classe majoritária; a CNN distribui melhor o acerto entre classes"* — não "a CNN é melhor em tudo".
3. **A interpretabilidade explica a polarização do ViT.** O ViT não aprende a atender as estruturas discriminativas (atenção difusa, bordas) → não distingue bem as classes sutis → "chuta" a majoritária. A interpretabilidade **explica** por que o balanced_acc do ViT é baixo apesar da accuracy alta. Evidência visual + métrica se reforçam.
4. **Swin é o meio-termo vencedor** — transformer (capacidade, atenção global) + viés local (janelas → eficiência em dados pequenos). Foca os ventrículos (como a CNN) E tem capacidade de transformer.
5. **Validação clínica:** Swin e ResNet focam os ventrículos (dilatação ventricular = marcador de atrofia em Alzheimer) — aprenderam sinal real, não atalho. O ViT não.

## Implicação para o TCC

- **Seção de discussão central:** "Transformers puros (ViT) vs híbridos (Swin) vs CNN (ResNet) em regime de dados limitados". Tese: com poucos pacientes, o viés indutivo importa — CNN e Swin > ViT puro, confirmado por métricas E interpretabilidade.
- **Contribuição reforçada:** não só aplicamos ViT/Swin ao OASIS com split por paciente (gap, F-0015), como mostramos *por que* o ViT puro fica atrás, com evidência visual.
- **Honestidade preservada:** reportar que o ViT ficou abaixo da CNN não é fracasso — é um achado alinhado à literatura e sustentado por interpretabilidade.

## Notas / armadilhas

- **Não cherry-pick.** OAS1_0316 ilustra bem, mas inspecionar mais casos antes de afirmar categoricamente; mostrar também onde os modelos erram.
- **Resolução dos mapas difere entre métodos:** Grad-CAM Swin (7×7), Grad-CAM ResNet (7×7), Attention Rollout ViT (14×14). A "difusão" do ViT não é só resolução — o Rollout tem MAIS resolução que o Grad-CAM e ainda assim espalha; isso reforça que é característica do modelo, não artefato.
- **McNemar das 3 arquiteturas:** predições do ResNet V2 re-geradas via `src/evaluation/eval_checkpoint.py` (treino foi antes da persistência). Completar a tabela de p-valores ResNet vs ViT vs Swin.
- Confirmar a leitura anatômica (ventrículos) com o orientador antes de afirmar no texto (lição F-0003).
