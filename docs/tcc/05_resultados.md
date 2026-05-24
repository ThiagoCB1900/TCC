# 5. Resultados

> Rascunho em texto corrido. Números conferidos contra `experiments/runs/` e os Findings citados. Marcadores `[ref: ...]` indicam a fonte — remover na versão final.

## 5.1 Análise exploratória

A análise exploratória confirmou as 86.437 fatias de 347 pacientes e o desbalanceamento severo entre as três classes [ref: F-0002]:

| Classe | Pacientes | Fatias | % |
|---|---:|---:|---:|
| Non Demented | 266 | 67.222 | 77,8 |
| Very Mild | 58 | 13.725 | 15,9 |
| Mild or Moderate | 23 | 5.490 | 6,4 |

A razão de ~12:1 entre a maior e a menor classe (em pacientes) motivou as escolhas de função de perda ponderada e de métricas balanceadas (Seções 4.5 e 4.8).

## 5.2 Linha de base convolucional (ResNet-50)

A primeira configuração (V1) sofreu **sobreajuste catastrófico**: a perda de treino colapsou cerca de 30× em quatro épocas, enquanto a perda de validação triplicou, com a melhor época sendo a primeira [ref: F-0013]. A revisão dos hiperparâmetros (V2) **controlou o sobreajuste** — a perda de treino passou a cair gradualmente ao longo de 16 épocas e a de validação permaneceu estável —, embora sem elevar o teto de desempenho [ref: F-0017]. A ablação confirmou empiricamente o valor da perda ponderada: o F1 da classe minoritária subiu de 0,17 (sem peso) para 0,40 (com peso), e a *balanced accuracy* de 0,50 para 0,62 [ref: F-0013]. Adotou-se a configuração V2 ponderada como linha de base oficial (treino estável; *balanced accuracy* de teste 0,587).

## 5.3 Vision Transformers (ViT-Base e Swin-T)

Com os hiperparâmetros V2, ambos os *transformers* voltaram a sobreajustar rapidamente (melhor época 2; perda de validação explodindo até ~3,0) [ref: F-0018]. A configuração V3, com regularização mais forte, **destravou ganho real** [ref: F-0019]: a melhor época do Swin-T avançou para a quarta, a perda de validação estabilizou-se (pico ~1,5) e a *balanced accuracy* de teste subiu de 0,594 para 0,616. O ViT-Base também melhorou de forma estatisticamente significativa (0,521 → 0,568; McNemar p≈0), mas permaneceu sobreajustando (melhor época 2), evidenciando que seus 85,8M de parâmetros são excessivos para 242 pacientes de treino.

## 5.4 Ablação de capacidade (ResNet-18)

Para testar a hipótese de que a ResNet-50 sobreajustaria por excesso de parâmetros, treinou-se a ResNet-18 (~metade dos parâmetros), mantendo transfer learning e metodologia idênticos. A hipótese foi **refutada**: a ResNet-18 obteve o pior desempenho entre todas as arquiteturas (*balanced accuracy* 0,501), significativamente inferior à ResNet-50 (McNemar p=2,8×10⁻⁶) [ref: F-0022]. O resultado confirma que o sobreajuste não era o gargalo da ResNet-50; reduzir capacidade apenas degradou o desempenho, sobretudo na classe minoritária (F1 de 0,323 para 0,208).

## 5.5 Comparação consolidada e significância estatística

A Tabela abaixo consolida os modelos no conjunto de teste (13.237 fatias, `split_v1`, *weighted cross-entropy*):

| Modelo | Params | Acurácia | **Balanced acc** | Macro-F1 | AUC | F1 (Mild+Mod) |
|---|---:|---:|---:|---:|---:|---:|
| **Swin-T V3** | 27,5M | 0,720 | **0,616** | **0,551** | **0,836** | **0,426** |
| ResNet-50 V2 | 23,5M | 0,681 | 0,587 | 0,513 | 0,804 | 0,323 |
| ViT-Base/16 V3 | 85,8M | 0,692 | 0,568 | 0,514 | 0,811 | 0,354 |
| ResNet-18 | 11,2M | 0,666 | 0,501 | 0,458 | 0,777 | 0,208 |

O **Swin-T V3 foi o melhor modelo** em todas as métricas. O teste de McNemar indicou diferenças estatisticamente significativas (p<0,05) em todos os pares de arquiteturas. Destaca-se uma inversão metodologicamente importante: o ViT-Base apresenta acurácia bruta superior à ResNet-50 (0,692 vs. 0,681), mas *balanced accuracy* inferior (0,568 vs. 0,587), por polarizar suas predições para a classe majoritária. Por isso, o McNemar — que mede acertos brutos — favorece o ViT no par ResNet-vs-ViT, conclusão que se inverte sob a métrica balanceada; reforça-se a necessidade de interpretar ambos em conjunto [ref: F-0021].

Como referência de contraste, o trabalho RanCom-ViT reporta 99,54% de acurácia no mesmo conjunto de dados, porém com particionamento por fatia [ref: F-0008]; a comparação direta de valores é inválida, pois mede fenômenos distintos (Seção 6).

## 5.6 Interpretabilidade

A análise qualitativa dos mapas, no mesmo paciente de demência (OAS1_0316), revelou convergência entre desempenho e foco anatômico [ref: F-0020, F-0021]:

- **ResNet-50 (Grad-CAM)** e **Swin-T (Grad-CAM)** concentraram a ativação nos **ventrículos laterais e região periventricular** — marcadores reais de atrofia em Alzheimer (dilatação ventricular). Foco limpo e clinicamente plausível.
- **ViT-Base (Attention Rollout)** apresentou **atenção difusa**, com focos em bordas e cantos da imagem, sem localizar as estruturas centrais.

Esse resultado é central para a discussão: a interpretabilidade **explica** o desempenho. O ViT puro, por não aprender a atender as estruturas discriminativas no regime de dados limitado, distingue mal as classes sutis e polariza para a majoritária — coerente com sua *balanced accuracy* inferior e com sua natureza *data-hungry*. Modelos com viés local (CNN e Swin) localizam o sinal clínico correto. Para as classes sutis (*Very Mild*, *Non Demented*), os mapas são mais difusos em todos os modelos, condizente com a dificuldade clínica intrínseca do estágio inicial. Registra-se a limitação de resolução do Grad-CAM no Swin (grade 7×7) [ref: F-0020].
