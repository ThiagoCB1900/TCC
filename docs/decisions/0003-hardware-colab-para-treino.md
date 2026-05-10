# ADR-0003 — Treino no Colab, dev local em CPU (RX6600 não viável)

- **Data:** 2026-05-10
- **Status:** Accepted
- **Decisores:** Thiago, Claude

## Contexto

Aluno tem GPU AMD Radeon RX 6600 em Windows. Treino de ViT-Base em ~86k slices é inviável em CPU (semanas). Precisamos definir onde rodar treino e dev.

## Alternativas consideradas

- **A — RX6600 + PyTorch DirectML** (Microsoft, suporte AMD em Windows). Funciona, mas: cobertura de operadores incompleta, performance inferior a CUDA equivalente, ViT/Swin frequentemente esbarram em ops não suportadas. Fora do mainstream de pesquisa.
- **B — RX6600 + ROCm em WSL2/Linux.** ROCm é a stack oficial AMD para deep learning, **mas só roda em Linux** e **a RX6600 não está oficialmente suportada** (suporte de consumo começa na RX 6800). Tentar é arriscar muito tempo de setup.
- **C — Colab (T4 grátis ou A100/L4 no Pro).** Padrão de pesquisa em deep learning quando não há acesso a cluster próprio. T4 tem ~16GB VRAM, suficiente para ViT-Base com batch moderado. A100 fecha treino completo em horas.
- **D — Outras nuvens (AWS/GCP/Lambda).** Custo significativo, curva de setup, fora do escopo de TCC.

## Decisão

Optamos por **C — Colab** como ambiente principal de treino e **CPU local** para EDA, dataloaders, smoke tests e geração de mapas de atenção pós-treino:

- EDA, manifesto, splits, dataloaders, e validação de pipeline rodam local com PyTorch CPU.
- Treino de ResNet-50, ViT-Base e Swin-T executa em notebooks Colab que importam o código do repositório via `git clone` direto do GitHub.
- Dataset OASIS é montado no Colab via Google Drive (upload one-shot) ou re-download do Kaggle dentro do notebook.
- Checkpoints salvos no Drive, métricas registradas em planilha simples (princípio operacional 5 do CLAUDE.md).

## Consequências

- Migração entre as duas máquinas do aluno fica trivial: ambas rodam o mesmo `requirements.txt` em `.venv/` local; o Colab é stateless por sessão e puxa código do GitHub.
- Não dependemos da RX6600 — fica disponível como bonus se o aluno quiser tentar DirectML em algum momento, mas não é caminho crítico.
- Limitação do Colab grátis: ~12h por sessão, GPU pode ser revogada. Mitigação: salvar checkpoints frequentemente; preferir Colab Pro se houver budget.
- ViT-Base com batch=32 a 224×224 cabe em T4 (~15GB ocupado). Swin-T idem. ResNet-50 caberia em batches maiores ainda.
- Reprodutibilidade: fixar seeds Python/NumPy/PyTorch, usar `torch.use_deterministic_algorithms(True)` quando viável, registrar versões de pacotes (já feito via `requirements.txt`).

## Referências

- CLAUDE.md, seção "Estado atual" e "Stack técnica"
- PyTorch Get Started: https://pytorch.org/get-started/locally/
