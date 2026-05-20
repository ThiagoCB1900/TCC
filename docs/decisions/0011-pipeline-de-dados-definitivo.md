# ADR-0011 — Pipeline de dados definitivo (consolida 0002/0004/0005/0006/0007)

- **Data:** 2026-05-17
- **Status:** Accepted (consolidador — não invalida ADRs anteriores, organiza-os em pipeline único)
- **Decisores:** Thiago, Claude

## Contexto

Antes de focar exclusivamente nos modelos (ViT-Base, Swin-T), precisamos **fechar o pipeline de dados** para não ter que refazer depois. F-0016 auditou as práticas dos 14 papers + best-practices externas (com fontes). Este ADR consolida tudo numa especificação única, marca o que é **fixo** vs **ablação opcional**, e enuncia as **regras de ouro** que nenhum experimento pode violar.

## Decisão — pipeline de dados (ordem de execução)

```
Para cada slice (JPG OASIS-Kaggle):
  1. Carregar JPG (RGB; R=G=B sintético — F-0003)
  2. [TRAIN ONLY] Augmentation (ver §Augmentation)
  3. Resize squash 224×224 (ViT-Base/ResNet) | 384×384 (Swin)   ← ADR-0004
  4. ToTensor float32 [0,1]
  5. Normalize ImageNet (mean/std padrão)                        ← ADR-0004
```

Split por paciente (ADR-0002) **precede tudo** — augmentation e cálculo de pesos acontecem só sobre os pacientes do fold de treino.

### Pré-processamento (FIXO)

- **Resize squash 224×224** (ViT-Base, ResNet-50) ou 384×384 (Swin-T). Mantido de ADR-0004.
- **Normalização ImageNet** (`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`). Mantido de ADR-0004.
- **RGB sintético mantido** (3 canais iguais) — compatibilidade com pesos pré-treinados. F-0003.
- **Skull stripping / registro / bias field**: **não fazemos** — já vêm prontos do dataset Kaggle (F-0001). Trabalhos como Vision-ViT-CNN fazem o pipeline FSL/ANTs completo no MRI bruto; nós herdamos a versão pré-processada.

### Balanceamento (FIXO)

- **Weighted CrossEntropyLoss `'balanced'`** (ADR-0007), pesos calculados **sobre o fold de treino**, aplicados **só na loss de treino**.
- **Validado independentemente**: Vision-ViT-CNN (trabalho mais comparável: ADNI, T1, 3-class, split por paciente) usa exatamente weighted loss (F-0016).
- **Val/test NUNCA balanceados** — dados intactos; desbalanceamento tratado nas **métricas** (balanced accuracy, macro-F1, AUC — ADR-0005), não nos dados.

### Augmentation (FIXO — V2, ADR-0006/ADR-0010)

- **Train only.** Aplicada dinamicamente por época, antes do resize.
- V2 atual: `RandomHorizontalFlip(0.5)` + `RandomRotation(±15°)` + `RandomAffine(translate 5%)` + `ColorJitter(0.2,0.2)`.
- **Nunca** em val/test (regra de ouro).

### Métricas (FIXO — ADR-0005)

- Primárias: macro-F1, balanced accuracy, AUC macro. McNemar para comparar modelos.
- Acurácia bruta só como secundária (comparabilidade com RanCom-ViT).

## Ablações opcionais (depois do baseline V2 — só se houver tempo/ganho)

Ordenadas por impacto esperado:

1. **Pipeline 2.5D** — 3 canais = slices vizinhos (n-1, n, n+1) do mesmo `(subject, session, mpr)` em vez de R=G=B. Ataca o desperdício dos canais sintéticos; reconhecido na literatura para datasets 2D pequenos + transfer learning (F-0016). **Maior potencial de ganho.** Vira "V3" se adotado.
2. **Enhancement de contraste** — CLAHE / histogram equalization antes da normalização. Vision-ViT-CNN usa; literatura reporta +contraste → +acurácia. Validar visualmente (não realçar ruído de fundo).
3. **Augmentation de intensidade** — blur/noise/ghosting leves (estilo TorchIO), p baixa. Vision-ViT-CNN usa.

Cada ablação só entra **uma de cada vez**, comparada contra V2 com McNemar, para isolar o efeito.

## Rejeitado (com justificativa)

| Técnica | Por que NÃO |
|---|---|
| Oversampling / SMOTE de slices | Poucos pacientes minoritários (17 train em mild_or_moderate) → repetir slices não cria diversidade real, só os mesmos pacientes mais vezes; e exigiria aplicação dentro do fold. Weighted loss é mais limpo e já validado. |
| Balancear/aumentar val ou test | **Leakage** + métricas irreais (viés até 0,34 AUC — Nature Sci Reports). Test deve refletir distribuição clínica real. |
| Normalização Nyúl / WhiteStripe | Dependem da distribuição de intensidade do MRI bruto; nosso dado é JPG (escala perdida). |
| Migrar para z-score | Quebraria compatibilidade com pesos ImageNet (transfer learning). Só consideraria se treinássemos do zero. |
| Test-Time Augmentation (TTA) | Técnica avançada fora do escopo; complica interpretabilidade. |

## Regras de ouro (NENHUM experimento viola)

1. **Split por paciente sempre** (ADR-0002). Augmentation, pesos e qualquer resampling só sobre o fold de treino.
2. **Augmentation e balanceamento: TRAIN ONLY.** Val/test intactos.
3. **Test reflete a distribuição real** — desbalanceamento tratado na métrica, nunca nos dados.
4. **Toda mudança no pipeline de dados re-valida visualmente** (lição F-0003) antes de treino real.
5. **Uma ablação por vez**, comparada com McNemar, para atribuição causal limpa.

## Consequências

- O pipeline de dados está **fechado** para o baseline e para ViT/Swin — podemos focar nos modelos sem retrabalho de dados.
- Conformidade do código atual **já verificada** (F-0016 §3): `build_dataloaders` (aug só train), `compute_class_weights` (sobre train fold), `evaluate` (métricas sem peso). Nenhuma correção necessária.
- As 3 ablações opcionais ficam como reserva de melhoria caso ViT/Swin também fiquem aquém — atacam dados, não arquitetura.
- Decisão revisitável: se a busca sistemática (recomendada em F-0015) revelar técnica dominante não considerada, reabrir este ADR.

## Referências

- Findings: F-0001 (dataset), F-0003 (RGB sintético), F-0005/F-0008/F-0014/F-0015 (trabalhos relacionados), F-0013 (baseline V1), **F-0016 (auditoria de práticas + best-practices, base deste ADR)**.
- ADRs consolidados: 0002 (split), 0004 (preproc), 0005 (métricas), 0006 (augmentation), 0007 (balanceamento), 0010 (V2).
- Fontes externas: imbalanced-learn common pitfalls; Nature Sci Reports 2024 (oversampling antes do CV); literatura de normalização MRI e 2.5D (ver F-0016).
