# F-0011 — Ler `Data/` direto do Drive (symlink) é gargalo severo de I/O no Colab

- **Data:** 2026-05-10
- **Status:** Confirmed (treino Colab interrompido por consumo de cota; correção pendente no notebook)
- **Categoria:** metodologia

## O que descobrimos

O notebook `01_resnet50_baseline_colab.ipynb` foi projetado para fazer o `Data/` apontar via symlink para `/content/drive/MyDrive/TCC/Data/`. A intenção era evitar uma cópia de 1,3 GB a cada sessão. Funcionou para EDA e smoke test (poucas leituras), mas **na primeira tentativa de treino real o aluno ficou sem cota de Colab gratuito antes do treino terminar nem 1 epoch** — porque cada `__getitem__` do DataLoader fez uma chamada FUSE individual ao Drive, e cada chamada tem latência alta.

### Custo de I/O extrapolado

- Dataset train: 60.329 imagens.
- Latência média estimada de uma leitura via FUSE Drive: 50-300 ms por arquivo (varia muito).
- I/O por epoch: 60.329 × ~150 ms = **~2,5 h só de leitura**, antes de qualquer compute.
- Com num_workers=2, **paraleliza pouco** porque Drive FUSE tem locks internos.
- Resultado: epoch que deveria levar 5-7 min em T4 leva **30-60+ min**.

Cota Colab gratuito é dada em "unidades computacionais" (compute units) — ficar segurando GPU enquanto espera I/O queima cota rápido. O aluno ficou sem cota antes da epoch 1 completar.

## Evidência

- Smoke test (240 imagens) rodou em ~2 min no T4 — não expôs o problema.
- Sessão Colab do aluno em 2026-05-10: "estava demorando muito" + desconexão por falta de unidades computacionais antes do treino completar.
- Padrão conhecido: comunidade Colab desencoraja ler datasets grandes (≥10k arquivos) direto do Drive durante treino. [Drive doc oficial recomenda copiar para disco local para datasets > 5 GB ou com muitos arquivos pequenos.]

## Implicação

**Correção obrigatória no `01_resnet50_baseline_colab.ipynb`** antes de tentar treino real no Colab de novo:

Em vez de fazer symlink direto, **copiar o `Data/` do Drive para o disco local SSD do Colab** (`/content/Data/`) UMA VEZ no início da sessão. Custo:

- Cópia inicial: ~3-5 min (one-shot).
- Leitura subsequente: SSD local ~10 MB/s aleatório → centenas de imagens por segundo → I/O deixa de ser gargalo.

Recomendado fazer a cópia a partir de **um `.zip` no Drive**, não da pasta com 86k arquivos:
1. No Drive: subir `Data.zip` (1,3 GB) UMA VEZ.
2. No notebook: `!cp /content/drive/MyDrive/TCC/Data.zip /content/Data.zip` (~30s) + `!unzip -q /content/Data.zip -d /content/` (~2 min).
3. Treino lê de `/content/Data/` em alta velocidade.

Outputs (`experiments/runs/`) podem continuar com symlink para o Drive — são poucos arquivos pequenos (config.yaml, history.json, checkpoint_best.pt), I/O total é baixo, persistência entre sessões compensa.

## Notas / armadilhas

- **Smoke test pequeno demais para detectar isso.** 240 imagens × 150 ms = 36 s — não dói. Precisa rodar batch de algumas centenas+ para detectar I/O gargalo. Lição: smoke test também deve **medir tempo médio por batch** e alertar se for absurdamente alto comparado a hardware esperado.
- **`num_workers > 0` no Colab + Drive não ajuda** como ajudaria em SSD. Drive FUSE serializa. Em SSD local, `num_workers=2` ou 4 acelera muito.
- **Pin_memory** também só ajuda em transferência CPU→GPU, irrelevante se gargalo é em CPU→Drive.
- Existem alternativas mais sofisticadas (WebDataset / tar streaming, `kaggle datasets download` direto) mas para o escopo do TCC a abordagem "unzip uma vez por sessão" é suficiente.
- **Cota Colab gratuito é renovada periodicamente** (a cada poucos dias). Se acabou, esperar 1-3 dias antes de tentar de novo. Sessão única em A100/L4 do Pro fecha o problema também.

## Ação imediata

1. Aluno gera `Data.zip` localmente: `Compress-Archive -Path Data -DestinationPath Data.zip` (ou via Windows Explorer).
2. Sobe `Data.zip` para `MyDrive/TCC/`.
3. Atualizar célula 4 (symlinks) e adicionar célula nova "copiar+unzip" antes do treino.
4. Re-tentar treino quando cota do Colab restaurar.

Como alternativa imediata para medir tempo real e seguir experimentando, criamos `notebooks/02_resnet50_local_cpu_timing.ipynb` para rodar 1 epoch completo em CPU local — mede o custo real sem depender do Colab.
