# ADR-0008 — Framework de treino: PyTorch puro com utilitários (torchmetrics + YAML), sem Lightning

- **Data:** 2026-05-10
- **Status:** Accepted
- **Decisores:** Thiago (delegou raciocínio), Claude

## Contexto

Para implementar o baseline ResNet-50 e os modelos ViT/Swin, precisamos definir o framework do loop de treino. Esta decisão impacta toda a fase 2-4 do plano de 60 dias (baseline → ViTs → interpretabilidade). Trocar de framework depois tem custo alto.

## Alternativas consideradas

### A — PyTorch puro (loop manual)
Loop explícito (`for batch in loader: optimizer.zero_grad(); ...`), métricas calculadas com `torchmetrics`, salvamento com `torch.save`, config em dataclass + YAML.

- **Prós:** zero magia; cada linha é explicável na banca; sem dependência nova; alinhado a princípio operacional 1 do CLAUDE.md ("end-to-end primeiro, qualidade depois"); auditável (princípio que estabelecemos após F-0003).
- **Contras:** mais boilerplate; fácil esquecer `model.eval()`, `torch.no_grad()`, `optimizer.zero_grad()` — mitigável com revisão cuidadosa e testes.

### B — PyTorch Lightning
Reorganizar treino em `LightningModule` com hooks (`training_step`, `validation_step`, `configure_optimizers`); `Trainer` cuida do loop, callbacks (EarlyStopping, ModelCheckpoint), reprodutibilidade (`seed_everything`), multi-GPU.

- **Prós:** padrão de pesquisa em DL acadêmico; reduz boilerplate; reprodutibilidade via `seed_everything`; multi-GPU "grátis".
- **Contras:** curva de aprendizado; dependência grande (PyTorch Lightning 2.x ~50 MB de pacotes); abstrações escondem o que acontece em cada step (em banca, "o framework gerencia" é resposta fraca); multi-GPU é benefício nulo aqui (Colab T4 é single-GPU); ~6 treinos no total não justifica o overhead.

### C — HuggingFace Trainer
- **Prós:** integra com HF Hub.
- **Contras:** mais voltado para NLP; pode ser overkill para timm + classificação simples; menos comum em pesquisa CV pura.

### D — Helpers do timm (`timm.utils`, `train.py` do próprio timm)
- **Prós:** alinhado com a biblioteca de modelos.
- **Contras:** `train.py` do timm é script CLI massivo, não biblioteca para integrar.

## Decisão

Optamos por **A — PyTorch puro** com utilitários já no `requirements.txt`:

- **Loop manual** em `src/training/loop.py` (`fit`, `train_one_epoch`, `evaluate`).
- **Métricas** em `src/evaluation/metrics.py` usando `torchmetrics` (já instalado) + `sklearn` para matriz de confusão e McNemar.
- **Config** em dataclass + YAML (`src/training/config.py`), serializável para auditoria entre máquinas.
- **Checkpoints** salvos manualmente em `torch.save(state_dict, path)`; melhor checkpoint preservado em `experiments/runs/<id>/checkpoint_best.pt` (ignorado pelo gitignore via `*.pt`).
- **History** salvo em `history.json` por época (loss train, loss val, métricas) — pequeno e versionável.
- **Reprodutibilidade** via fixação manual de seeds (`torch.manual_seed`, `random.seed`, `np.random.seed`) na entrada do script.
- **Sem `seed_everything`, sem `Trainer`, sem `LightningModule`.**

Para os ViT/Swin (semanas 3-4), o mesmo loop serve — basta trocar o `model = build_model(name)`. Não há ganho de Lightning para modelos diferentes desde que a interface seja a mesma.

### Hiperparâmetros do baseline ResNet-50 (referência inicial, ajustável)

| Hiperparâmetro | Valor | Justificativa |
|---|---|---|
| Optimizer | AdamW | Padrão atual para fine-tuning ViT/CNN modernas; bem-comportado |
| LR | 1e-4 | Fine-tuning ImageNet típico (10× menor que treino do zero) |
| Weight decay | 0.05 | Default timm para fine-tune |
| Batch size | 32 (treino), 64 (eval) | T4 16GB cabe; eval maior porque não há gradiente |
| Epochs (máximo) | 20 | Histórico timm/HF para fine-tune; early stopping deve cortar antes |
| Early stopping | patience=3 epochs sem melhora em **balanced accuracy de val** | Robusto a desbalanceamento; alinhado a ADR-0005 |
| LR scheduler | Cosine annealing | Convergência mais suave que reduce-on-plateau; padrão em ViT/timm |
| Grad clipping | `max_norm=1.0` | Padrão; previne explosion no início do treino com weighted CE |
| Warmup | 1 epoch linear | Estabiliza início; alinhado a ADR-0007 (warmup previne instabilidade da weighted loss) |
| Mixed precision | Off em CPU (smoke test); on em GPU Colab via `torch.cuda.amp.autocast` | 2× speedup sem perda de qualidade na maioria dos casos |
| Seeds | `torch.manual_seed(42)`, `random.seed(42)`, `np.random.seed(42)` | Mesmo seed do split (ADR-0002) por simplicidade |

### Estrutura de pastas

```
src/
├── models/
│   ├── __init__.py
│   └── resnet.py              # build_resnet50(num_classes, pretrained)
├── training/
│   ├── __init__.py
│   ├── config.py              # @dataclass TrainConfig + to_yaml/from_yaml
│   ├── loop.py                # fit/train_one_epoch/evaluate
│   └── run.py                 # CLI entrypoint
└── evaluation/
    ├── __init__.py
    └── metrics.py             # ClassificationMetrics, mcnemar_test

experiments/runs/
└── <YYYYMMDD_HHMMSS>_resnet50_class3_weighted/
    ├── config.yaml
    ├── train.log
    ├── checkpoint_best.pt     # .gitignored
    ├── history.json
    └── final_test_metrics.json
```

## Consequências

- **Toda a lógica é auditável.** Em banca, qualquer detalhe pode ser apontado no código.
- **Reprodutibilidade exige disciplina manual.** Mitigado por seeds fixas, config YAML salvo por run, history JSON.
- **Migração para Lightning depois é possível** se o projeto crescer (improvável no escopo de TCC). O loop em PyTorch puro mapeia 1:1 para LightningModule.
- **Smoke test sempre antes de treino real.** Padrão estabelecido: rodar 2 epochs com poucos batches em CPU local para validar fluxo antes de queimar tempo de Colab.
- **Loss curve + métricas por época são auditadas visualmente** após cada smoke test e após cada run real, antes de declarar sucesso. Aderência à lição F-0003.

## Referências

- Findings: F-0008 (RanCom-ViT — comparativo), F-0009 (literatura)
- ADRs anteriores: ADR-0001 (classes), ADR-0002 (split), ADR-0004 (pré-processamento), ADR-0005 (métricas), ADR-0006 (augmentation), ADR-0007 (weighted CE)
- timm fine-tuning guide: https://huggingface.co/docs/timm
- PyTorch best practices for reproducibility: https://pytorch.org/docs/stable/notes/randomness.html
