# TCC — Vision Transformers para Classificação de Alzheimer

> **Para o Claude Code:** este arquivo é a fonte da verdade do projeto. É lido automaticamente em toda sessão. Como o aluno trabalha em **dois computadores**, mantenha-o atualizado a cada decisão relevante (modelo, hiperparâmetro, resultado, mudança de escopo). A memória local de Claude Code não sincroniza entre máquinas — este arquivo, sim.

---

## Identificação

- **Aluno:** Thiago Guilherme Aguiar (`thiagoguilherme123g@gmail.com`)
- **Curso:** Ciências da Computação · UECE
- **Tema:** Vision Transformers para Classificação de Alzheimer
- **Dataset:** OASIS-1 (já obtido pelo aluno)
- **Início do plano de 60 dias:** 2026-05-07
- **Entrega da cópia (orientador):** fim da semana 4 (~2026-06-03)
- **Defesa prevista:** fim da semana 8 (~2026-07-06)

## Estado atual

- **Fase:** Semana 1 · Fundação · início da implementação
- **Estrutura de pastas criada:** sim (ver "Layout do repositório" abaixo)
- **Próximo passo imediato:** ações do Dia 1 — solicitar acesso OASIS, reunião com orientador, iniciar leitura do paper ViT (Dosovitskiy et al., 2020).

## Decisões obrigatórias (revisar até o fim da semana 1)

| Decisão | Recomendação inicial | Status |
|---|---|---|
| Versão do OASIS | OASIS-1 | a confirmar com orientador |
| Tipo de classificação | Binária (CDR=0 vs CDR>0) | a confirmar |
| Modelos comparados | ViT-Base/16 + Swin-T | a confirmar |
| Baseline obrigatório | ResNet-50 com pesos ImageNet | fixo |

## Regras críticas — NÃO violar

1. **Split por paciente (subject ID), nunca por slice.** Dividir slices aleatoriamente faz com que o mesmo paciente apareça em treino e teste, inflando a acurácia artificialmente e invalidando o trabalho.
2. **End-to-end primeiro, qualidade depois.** Pipeline completo (dados → modelo → métricas) deve rodar até o fim da semana 2, mesmo com performance ruim.
3. **Nunca cortar:** baseline ResNet-50, split por paciente, interpretabilidade básica (Attention Rollout). São os três pilares que diferenciam o trabalho.
4. **Validar manualmente** todo código de pré-processamento, split e métricas — o aluno precisa defender cada decisão na banca.

## Ordem de corte se houver atraso

1. Swin Transformer (mantém ViT-Base + ResNet)
2. Experimento de transfer learning sem pré-treino (mantém só fine-tuning ImageNet)
3. GradCAM no Swin (mantém só Attention Rollout no ViT)

## Stack técnica

- **Linguagem:** Python 3
- **Frameworks:** PyTorch, [timm](https://github.com/huggingface/pytorch-image-models) (modelos), [MONAI](https://monai.io/) (utilidades para imagem médica)
- **Pré-processamento:** FSL BET (skull stripping) ou usar pré-processamento já incluído no OASIS
- **Treino:** Colab Pro ou GPU local
- **Tracking de experimentos:** planilha simples (data, modelo, hiperparâmetros, métricas, observações). Nada mais sofisticado é necessário no escopo do TCC.

## Layout do repositório

```
TCC/
├── data/                    # OASIS bruto — NÃO commitar (ver .gitignore)
├── notebooks/               # exploração inicial, EDA
├── src/
│   ├── data/                # preprocessing, splits, dataloaders
│   ├── models/              # ViT, Swin, ResNet
│   ├── training/            # loops de treino
│   ├── interpretability/    # attention rollout, gradcam
│   └── evaluation/          # métricas, statistical tests (McNemar)
├── experiments/             # configs e logs de cada run
├── results/                 # tabelas, figuras, attention maps
├── docs/                    # papers lidos, fichamentos, draft do TCC
├── CLAUDE.md                # este arquivo — contexto para Claude Code
└── README.md
```

## Cronograma resumido (60 dias)

- **Sem 1 (dias 1-7):** fundação — leituras (ViT, Swin, survey Shamshad), ambiente, repositório, decisões fechadas com orientador.
- **Sem 2 (dias 8-14):** dados + baseline — download OASIS, EDA, pré-processamento, split por paciente, ResNet-50 baseline rodando.
- **Sem 3 (dias 15-21):** ViTs — fine-tuning ViT-Base e Swin-T, comparação com baseline, McNemar.
- **Sem 4 (dias 22-30):** interpretabilidade + entrega da cópia — Attention Rollout, GradCAM, análise qualitativa, entrega ao orientador no dia 30.
- **Sem 5-6 (dias 31-45):** refinamento — incorporar feedback, experimentos extras, escrita de resultados/discussão.
- **Sem 7-8 (dias 46-60):** fechamento — conclusão, revisões, slides (começar até dia 50), ensaios da defesa, defesa.

## Princípios operacionais

1. **End-to-end primeiro, qualidade depois.**
2. **Comece a escrever cedo** — introdução já na semana 2.
3. **Versione tudo desde o dia 1** (Git para código; planilha para experimentos).
4. **Reuniões semanais com orientador são inegociáveis** — enviar resumo escrito antes.
5. **Reserve buffer mental** — algo vai dar errado (GPU, OASIS, convergência); folga deliberada existe, não a queime no começo.

## Como manter este arquivo (instruções para o Claude Code)

- Sempre que houver decisão tomada, modelo escolhido, hiperparâmetro fixado, resultado relevante, ou mudança de escopo, **atualize a seção correspondente** deste arquivo.
- Atualize "Estado atual" com a fase atual e o próximo passo imediato sempre que avançar.
- Mantenha a tabela de "Decisões obrigatórias" refletindo o que já foi fechado vs. ainda em aberto.
- Adicione uma seção "Resultados consolidados" assim que houver primeiras métricas (ResNet baseline, ViT, Swin) — formato tabela.
- Adicione uma seção "Gotchas encontrados" se aparecerem armadilhas técnicas (ex.: bug em algum split, problema de convergência, cuidado com formato NIfTI).
- Não duplicar o checklist completo de 60 dias — ele está no PDF original do plano. Aqui mantemos só o resumo e o estado.

## Referências de papers (a consultar em src/docs/)

- Dosovitskiy et al., 2020 — *An Image is Worth 16×16 Words* (ViT)
- Liu et al., 2021 — *Swin Transformer*
- Shamshad et al. — survey de transformers em imagem médica
- He et al., 2015 — ResNet (baseline)
- Selvaraju et al., 2017 — Grad-CAM
- Abnar & Zuidema, 2020 — Attention Rollout
