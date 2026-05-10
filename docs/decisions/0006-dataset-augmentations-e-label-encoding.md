# ADR-0006 — Dataset PyTorch: augmentations leves e label encoding por severidade

- **Data:** 2026-05-10
- **Status:** Accepted
- **Decisores:** Thiago, Claude

## Contexto

ADR-0004 já fixou o pré-processamento básico (resize squash 224×224, manter RGB sintético, normalização ImageNet). Faltam duas decisões metodológicas para o `Dataset` PyTorch:

1. **Quais augmentations** aplicar no split de treino (e nenhuma no val/test).
2. **Qual ordem inteira** atribuir às classes textuais para o `target` do classificador (impacta interpretação de matriz de confusão e ordem das colunas em métricas por classe).

## Alternativas consideradas

### Augmentations

- **A — Sem augmentation.** Simplicidade, mas aumenta o risco de overfit no `mild_or_moderate` (23 pacientes).
- **B — Flip horizontal apenas.** Dobra os dados efetivos, é regularização barata, modelo aprende invariância L/R. **Risco:** Alzheimer pode iniciar assimetricamente (atrofia hipocampal frequentemente é mais pronunciada em um lado), e flip pode mascarar essa pista.
- **C — Pacote leve médico-realista:** flip horizontal + rotação leve (±5°) + jitter de brilho/contraste pequeno. Mais regularização, ainda anatomicamente plausível.
- **D — Pacote agressivo (translation, zoom, elastic, cutout, etc.).** Comum em ImageNet, mas em neuroimagem pode introduzir distorções não-fisiológicas e prejudicar a interpretabilidade.

### Label encoding (mapeamento string → int)

- **a — Ordem alfabética** (`mild_or_moderate=0`, `non_demented=1`, `very_mild=2`). Padrão de muitas bibliotecas (`LabelEncoder` do sklearn, `flow_from_directory`). **Não interpretável**: `0=mild_or_moderate, 1=non_demented` viola a intuição de "não-doente é o estado base".
- **b — Ordem por severidade clínica**: `non_demented=0`, `very_mild=1`, `mild_or_moderate=2`. Matriz de confusão fica intuitiva (estados saudáveis no canto superior esquerdo, severidade aumenta ao longo da diagonal).

## Decisão

**Augmentations — Opção C** (pacote leve médico-realista) só no split de treino:

- `RandomHorizontalFlip(p=0.5)` — anatomicamente justificado pela aproximada simetria sagital do cérebro normal. Reconhecemos o risco residual (assimetria de atrofia em Alzheimer pode ser informação clínica), mas: (i) pesos ImageNet pré-treinados já aprenderam invariância a flip, (ii) é a augmentation universalmente usada em papers de classificação Alzheimer 2D MRI, (iii) o test set não é flippado, então qualquer dependência aprendida a partir do flip não corrompe a métrica final.
- `RandomRotation(degrees=5)` — pequenas variações de posicionamento entre aquisições são realistas (paciente não fica perfeitamente alinhado). Usar `interpolation=BILINEAR` e `fill=0` (preto, consistente com fundo da imagem).
- `ColorJitter(brightness=0.1, contrast=0.1)` — variações sutis de intensidade simulam diferenças de protocolo/scanner. Não usar saturação ou matiz (faria sentido em foto natural; aqui R==G==B, mexer em saturação distorce o sinal).
- **NUNCA** `RandomVerticalFlip` — cabeça invertida não ocorre clinicamente.
- **NUNCA** augmentation no val ou test — viola a regra de avaliação determinística.

**Label encoding — Opção b** (ordem por severidade clínica). Para os dois esquemas:

| class_3 | int | class_binary | int |
|---|---:|---|---:|
| non_demented | 0 | non_demented | 0 |
| very_mild | 1 | demented | 1 |
| mild_or_moderate | 2 | | |

Ordem **fixa, hardcoded** no Dataset (não inferida automaticamente do manifesto). O `Dataset` expõe atributo `class_to_idx` e seu inverso `idx_to_class` para inspeção e plotagem.

## Consequências

- Augmentations são aplicadas **dentro do dataloader, sob demanda em cada época**. Não geram arquivos extras no disco. Não vazam para val/test.
- Matriz de confusão fica intuitiva: linha 0 = `non_demented` real, coluna 0 = predito como `non_demented`. Diagonal aumenta em severidade, off-diagonal mostra confusões interessantes (ex: `very_mild` confundido com `non_demented` é diferente clinicamente de `mild_or_moderate` confundido com `non_demented` — o segundo é mais grave).
- Reprodutibilidade: augmentations dependem do seed do DataLoader / generator. Para experimentos comparativos exatos, fixar `torch.manual_seed` antes do treino. Para `evaluate` em test set não há randomness (sem augmentation).
- A decisão sobre `RandomHorizontalFlip` é revisitável: se a fase de interpretabilidade (semana 4) sugerir que o modelo está usando assimetria como pista clinicamente significativa, podemos remover o flip e re-treinar. Documentar no log de experimentos.
- Pacote agressivo (Opção D) fica como ablação opcional na semana 5-6 se houver tempo.

## Referências

- ADR-0001 (3 classes) — define quais classes textuais existem
- ADR-0002 (split por paciente) — define que augmentation acontece por fold, não global
- ADR-0004 (resize 224×224 + ImageNet norm) — pré-processamento determinístico antes de augmentation
- F-0003 (single axial layout) — confirma que a imagem é um corte axial, justificando flip horizontal seguro
- torchvision.transforms.v2: https://pytorch.org/vision/stable/transforms.html
- Discussão sobre assimetria em Alzheimer: Thompson et al., *Tracking Alzheimer's disease*, 2007
