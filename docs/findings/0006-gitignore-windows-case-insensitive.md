# F-0006 — `.gitignore` `Data/` sem âncora ignora `src/data/` no Windows

- **Data:** 2026-05-10
- **Status:** Confirmed (corrigido no momento da descoberta)
- **Categoria:** metodologia

## O que descobrimos

A regra `Data/` no `.gitignore` (sem barra inicial) **casa com qualquer pasta `Data` ou `data` em qualquer profundidade** do repositório, e não apenas com `Data/` na raiz.

Em Windows, o git roda com `core.ignorecase=true` por padrão, então `Data/` casa com `data/` também. Resultado: a regra criada para ignorar o dataset OASIS na raiz (`Data/`) estava **silenciosamente ignorando os scripts em `src/data/`** (`eda.py`, `inspect_image_layout.py`, `__init__.py`).

A correção é **ancorar à raiz** com barra inicial: `/Data/` e `/data/`.

## Evidência

```
$ git check-ignore -v src/data/eda.py
.gitignore:3:Data/    src/data/eda.py        # ← regra Data/ casava com src/data/
```

Após correção:

```
$ git check-ignore -v src/data/eda.py
exit=1                                        # ← não ignorado mais

$ git check-ignore -v Data/
.gitignore:4:/Data/    Data/                  # ← raiz continua ignorada
```

## Implicação

- **Quase-acidente sério:** os scripts de EDA (`src/data/eda.py`, `inspect_image_layout.py`) não estavam aparecendo em `git status` como untracked. Se o commit fosse feito sem essa auditoria, a fundação do pipeline iria para o repositório **sem o código que a gerou**, quebrando reprodutibilidade entre as duas máquinas do aluno.
- A descoberta foi feita por acaso ao listar os untracked com `git status -u` em vez de `git status` simples — `git status` agrupava tudo sob `src/__init__.py` (o único untracked dentro de `src/` aos olhos do git porque o resto estava sendo ignorado).

## Notas / armadilhas

- **Regra geral:** sempre ancore com `/` regras de gitignore que devem casar apenas com diretórios da raiz. Sem âncora, casa em qualquer profundidade.
- **Sempre rodar `git status -u` antes do primeiro commit de uma sessão** — assim untracked dirs aparecem expandidos arquivo por arquivo, expondo silenciosos.
- Adicional: rodar `git check-ignore -v <caminho>` quando suspeitar que algo deveria estar versionado mas não aparece.
- Mesma armadilha pode ocorrer com qualquer regra "genérica" no gitignore (`logs/`, `tmp/`, `cache/`, etc.). Auditar todas.

## Re-ocorrência (2026-05-10) — mesma classe de bug com `runs/`

Detectado durante o commit do baseline ResNet-50: a regra `runs/` (sem âncora, originalmente para ignorar `wandb/runs/` etc.) também casava com `experiments/runs/` — o que **silenciosamente ignorava todos os artefatos das runs de treino reais** (config.yaml, history.json, final_test_metrics.json), que **devem ser versionados** para auditoria entre máquinas.

Correção: removida a regra genérica `runs/`; mantidas regras específicas com âncora (`/experiments/runs/*smoke*/` para smokes) e regras de arquivo (`*.pt`, `*.log`). Comentário no `.gitignore` documenta a armadilha.

**Lição reforçada:** auditar todas as regras genéricas. Cada vez que uma nova pasta nomeada existe no projeto (ex: `runs/`, `data/`, `logs/`), revisar se há regra que pode casar inadvertidamente.
