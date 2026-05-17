# ADR-0009 — Kaggle Notebooks como ambiente primário de treino

- **Data:** 2026-05-10
- **Status:** Accepted · Substitui ADR-0003 (que fica como "Superseded by ADR-0009")
- **Decisores:** Thiago (propôs), Claude (analisou e implementou)

## Contexto

ADR-0003 (2026-05-10, manhã) fixou Colab gratuito como ambiente de treino, com CPU local apenas para EDA/dev. Na tarde do mesmo dia, a primeira tentativa de treino no Colab **falhou antes de completar 1 epoch** — F-0011 diagnosticou que o gargalo era I/O do Drive (cada `__getitem__` = chamada FUSE lenta). O Colab ficou queimando cota com GPU idle esperando o Drive.

O aluno sugeriu **Kaggle Notebooks** como alternativa. Análise técnica em F-0012 confirma: Kaggle hospeda o dataset (`ninadaithal/imagesoasis`) nativamente em SSD local da sessão, eliminando o problema de I/O por design.

## Alternativas consideradas

### A — Manter Colab grátis, corrigindo F-0011 (Data.zip → /content/)
- **Prós:** infraestrutura já desenvolvida (notebook 01); Drive como persistência de outputs.
- **Contras:** 1,3 GB de cópia a cada sessão (3-5 min); ainda exige upload de `Data.zip` para o Drive; sessões mais imprevisíveis que Kaggle; cota "compute units" opaca.

### B — Kaggle Notebooks
- **Prós:** dataset pronto (zero upload); SSD local elimina F-0011; cota 30h/semana de GPU explícita; notebook salva versões com inputs+código+outputs (reprodutibilidade reforçada); internet pra `git clone` do nosso repo.
- **Contras:** internet precisa ser habilitada manualmente; outputs em `/kaggle/working/` precisam ser commitados/baixados manualmente; sessões limitadas a 9h (vs 12h do Colab).

### C — Colab Pro (R$ 50-60/mês, A100/L4)
- **Prós:** GPU muito superior, fecha treino em ~30 min.
- **Contras:** custo; não resolve problema do dataset (precisa upload Drive ou Kaggle API).

### D — CPU local (já temos notebook 02 medindo timing)
- **Contras:** notebook 02 está medindo, mas estimativa anterior é ~5-12h por epoch — totalmente inviável para 4-6 runs.

## Decisão

Optamos por **B — Kaggle Notebooks como ambiente primário** de treino:

- Treinos reais (ResNet-50 baseline, ablações, ViT-Base, Swin-T) vão para Kaggle Notebooks.
- `notebooks/03_resnet50_baseline_kaggle.ipynb` é o ponto de entrada.
- Outputs (`history.json`, `final_test_metrics.json`, `config.yaml`) são commitados de volta no repo via download manual + git push local. Checkpoints (`*.pt`) ficam no notebook salvo do Kaggle (não vão pro repo por causa do `.gitignore`).

ADR-0003 **fica marcado como "Superseded by ADR-0009"** mas não é apagado: continua documentando o raciocínio inicial e Colab continua como fallback viável (notebook 01 será corrigido conforme F-0011 para esse cenário). CPU local (notebook 02) permanece para timing e smoke tests.

## Hierarquia de ambientes pós-ADR-0009

| Prioridade | Ambiente | Quando usar |
|---|---|---|
| 1 (primário) | **Kaggle Notebooks** | Treino de qualquer modelo do plano |
| 2 (backup) | **Colab gratuito** | Se Kaggle indisponível; precisa do fix de Data.zip (F-0011) |
| 3 (escala) | **Colab Pro** | Se cronograma apertar (~1 mês de Pro fecha tudo em 1-2 dias) |
| Sempre | **CPU local + VSCode** | Edição, EDA, smoke, escrita, análise pós-treino |

## Consequências

- **Reduz risco do cronograma:** problema do F-0011 deixa de bloquear o aluno.
- **Persistência manual:** outputs precisam ser baixados do Kaggle e commitados — adicionar passo no fluxo de cada run.
- **`git_commit` no `config.yaml`** continua sendo a fonte de verdade de qual versão do código gerou cada resultado; "version" do notebook Kaggle é redundância adicional.
- **Notebook 01 (Colab) fica defasado e precisa ser atualizado** com o fix do F-0011 antes de poder ser usado como backup confiável. Por ora, ganha banner "ver notebook 03 primeiro".
- **Decisão revisitável:** se Kaggle Notebooks mudar políticas (cota, GPU, dataset) durante o TCC, voltamos ao Colab Pro como plano B.

## Referências

- Findings: F-0011 (problema do Colab+Drive), F-0012 (Kaggle Notebooks como solução)
- ADR substituído: ADR-0003
- Notebooks: 01 (Colab — precisa update), 02 (CPU local timing), 03 (Kaggle, novo primário)
