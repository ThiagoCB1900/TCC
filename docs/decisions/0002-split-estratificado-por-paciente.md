# ADR-0002 — Split estratificado por subject ID, nunca por slice

- **Data:** 2026-05-10
- **Status:** Accepted
- **Decisores:** Thiago, Claude

## Contexto

Cada paciente do OASIS-1 contribui em média ~244 slices ao dataset (4 aquisições mpr × 61 slices axiais centrais; 19 controles têm o dobro por causa do reescaneamento MR2 — ver F-0004). Um split aleatório por arquivo coloca o **mesmo paciente** em treino e teste, fazendo o modelo aprender particularidades anatômicas e ruído de aquisição daquele indivíduo, em vez de generalizar para a doença.

A literatura tem inúmeros exemplos de papers que reportam acurácia ≥99% em Alzheimer/MRI exatamente por essa falha (ver F-0005 sobre o notebook CNN comparativo).

## Alternativas consideradas

- **A — Split aleatório por arquivo** (`flow_from_directory`, `train_test_split` ingênuo). Trivial de implementar, é o que o notebook CNN comparativo fez. **Inválido**.
- **B — Split estratificado por paciente, com replicatas mpr/MR2 inseparáveis.** Agrupa todos os arquivos de um mesmo `subject` no mesmo split. Garante zero vazamento. Requer estratificação manual para preservar proporção de classes em cada split.
- **C — Cross-validation k-fold por paciente.** Mais robusto estatisticamente; mais caro computacionalmente (treina k vezes). Pode ser overkill para um TCC com prazo de 60 dias.

## Decisão

Optamos por **B — split estratificado por subject ID** com proporções 70/15/15 (train/val/test). Implementação em `src/data/splits.py`:

- Agrupar arquivos por `subject`.
- Estratificar a alocação de subjects (não arquivos) por classe usando `class_3` (ver ADR-0001).
- Seed fixa salva no `splits.json`.
- Validação automática: assert que conjunto de subjects em train, val e test são disjuntos.
- Persistir a saída como `experiments/splits/split_v1.json` com listas de subject IDs por fold, para reprodutibilidade entre as duas máquinas do aluno.

K-fold (alternativa C) fica como evolução futura se o orientador pedir, mas **70/15/15 single split é o suficiente para um TCC** dado o tempo limitado e o uso de McNemar para comparar modelos no mesmo test set.

## Consequências

- Esta é a regra crítica nº 1 do CLAUDE.md ("Split por paciente, nunca por slice"). Qualquer commit que reverter isso quebra a tese inteira.
- Test set terá ~52 pacientes (15% de 347), distribuídos pelas 3 classes — suficiente para macro-F1 e McNemar.
- Dado os 23 pacientes em `mild_or_moderate`, o test set tem ~3-4 nessa classe → resultados por classe terão variância alta. Reportar com intervalos de confiança (bootstrap).
- Augmentation (se aplicado) só pode ser feito **dentro do dataloader, sobre o split de treino**. Nunca sobre o dataset inteiro antes do split.

## Referências

- Findings: F-0002, F-0004, F-0005
- CLAUDE.md, "Regras críticas — NÃO violar", item 1
- Marcus et al. 2007 (paper original OASIS-1)

## Implementação concreta (2026-05-10)

Implementado em `src/data/splits.py`. Saída em `experiments/splits/split_v1.json`. Comando reprodutível:

```powershell
python -m src.data.splits --seed 42
```

**Resultado para o split v1 (seed=42, stratify=class_3, 70/15/15):**

| Fold | n_subjects | n_slices | non_demented | very_mild | mild_or_moderate |
|---|---:|---:|---:|---:|---:|
| train | 242 | 60.329 | 185 | 40 | 17 |
| val | 52 | 12.871 | 40 | 9 | 3 |
| test | 53 | 13.237 | 41 | 9 | 3 |
| **Total** | **347** | **86.437** | **266** | **58** | **23** |

Proporções por fold ficam dentro de ~1% da distribuição global em todas as classes. Asserts de disjunção entre folds e cobertura de classes (todas as classes em todos os folds) passaram. Reprodutibilidade verificada: re-rodar com seed=42 produz o mesmo conjunto de subjects por fold (apenas o campo `created_at` muda).
