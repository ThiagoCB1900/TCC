# Estrutura do TCC e mapa de evidências

> Esqueleto-mestre da monografia. Cada seção lista **o que escrever** e **qual ADR/Finding/run sustenta** — assim a escrita é só "traduzir" evidência já produzida em texto corrido. Markdown aqui é rascunho versionável; o texto final vai para o template ABNT da UECE (Word ou abnTeX2).

**Título provisório:** *Vision Transformers para Classificação de Estágios de Alzheimer em Ressonância Magnética: um estudo com rigor metodológico no OASIS-1*

**Argumento central (fio condutor):** com split por paciente e métricas balanceadas (rigor que boa parte da literatura no mesmo dataset não adota), as métricas honestas são modestas (~0,62 balanced acc), e — confirmado por métricas + interpretabilidade + teoria — modelos com viés local (CNN, Swin) superam o ViT puro em regime de dados pequenos.

---

## 1. Introdução  → escrever por último (depende do resto)
- Contexto: Alzheimer, importância do diagnóstico precoce, MRI. (Fundamentação)
- Problema: classificação automática de estágios; desafios (desbalanceamento, dados limitados, leakage na literatura).
- **Lacuna/contribuição:** ViT + OASIS-1 + split por paciente é interseção vazia entre os trabalhos auditados (**F-0015**); a maioria no mesmo dataset usa split por slice e infla métricas (**F-0008**).
- Objetivos: geral (comparar CNN vs ViT vs Swin com rigor + interpretabilidade) e específicos.
- Contribuições: (i) pipeline rigoroso reprodutível; (ii) comparação honesta de 4 arquiteturas via McNemar; (iii) interpretabilidade que explica o ranking; (iv) crítica metodológica fundamentada.

## 2. Fundamentação Teórica
- Doença de Alzheimer e biomarcadores em MRI (atrofia, ventrículos, hipocampo). → Gotchas/F-0020 (ventrículos), papers Alzheimer.
- Neuroimagem: OASIS-1, MRI T1, fatias axiais. → **F-0001, F-0002**, gotchas do CLAUDE.md.
- CNNs e ResNet (He 2016). → paper ResNet.
- Vision Transformers: ViT (Dosovitskiy 2020), atenção, natureza *data-hungry*. → papers ViT, **F-0021**.
- Swin Transformer (Liu 2021): janelas, viés local. → paper Swin.
- Interpretabilidade: Grad-CAM (Selvaraju 2017), Attention Rollout (Abnar 2020), Chefer 2021. → **ADR-0014**, papers.
- Métricas para dados desbalanceados: balanced acc, macro-F1, AUC, McNemar. → **ADR-0005, F-0021**.

## 3. Trabalhos Relacionados  → quase pronto (F-0014, F-0015)
- Tabela-mestre de split por paciente (**F-0014**): quem faz, quem não faz, quem não explicita.
- RanCom-ViT (mesmo dataset, split por slice, 99,54%): oponente metodológico (**F-0008**).
- ADNI + split por paciente (Leveraging Swin, VGG-TSwinformer, Vision-ViT-CNN): boa prática (**F-0014**).
- Joint Transformer: zona cinzenta (split por volume, leakage longitudinal) (**F-0014**).
- Dataset uraninjo / notebook CNN: split por paciente impossível (**F-0005, F-0014**).
- Mapa de posicionamento e o gap (**F-0015**). Meta-análise QUADAS-2 (31% sem detalhe de seleção).
- *(Pendente: busca sistemática Scholar/PubMed para blindar o gap — recomendado em F-0015.)*

## 4. Metodologia  → ESCRITA em `04_metodologia.md` (a mais pronta)
- Dataset OASIS-Kaggle: estrutura, nomenclatura, EDA. → **F-0001, F-0002, F-0003**, ADR-0001.
- Esquema de 3 classes. → **ADR-0001**.
- Split estratificado por paciente. → **ADR-0002** (regra de ouro).
- Pré-processamento: resize 224 squash, ImageNet norm, RGB sintético. → **ADR-0004, F-0003**.
- Pipeline de dados + regras de ouro (aug/balance só train). → **ADR-0011, F-0016**.
- Balanceamento: weighted CE. → **ADR-0007**.
- Augmentation. → **ADR-0006**.
- Modelos: ResNet-50/18, ViT-Base, Swin-T (timm). → **ADR-0012, ADR-0008**.
- Treino: PyTorch puro, hiperparâmetros V2/V3. → **ADR-0008, ADR-0010, ADR-0013**.
- Métricas + McNemar. → **ADR-0005**.
- Interpretabilidade. → **ADR-0014**.
- Ambiente (Kaggle). → **ADR-0009, F-0011, F-0012**.

## 5. Resultados  → ESCRITA em `05_resultados.md`
- EDA (distribuição, desbalanceamento). → results/eda/, **F-0002**.
- Baseline ResNet-50: V1 overfit → V2 controlado. → **F-0013, F-0017**.
- Transformers: V2 overfit → V3 destravou. → **F-0018, F-0019**.
- Ablação de capacidade (ResNet-18). → **F-0022**.
- Tabela consolidada das 6 runs + McNemar das 4 arquiteturas. → CLAUDE.md "Resultados consolidados", **F-0021, F-0022**.
- Interpretabilidade: Swin/ResNet focam ventrículos; ViT difuso. → **F-0020, F-0021**, results/interpretability/.

## 6. Discussão
- Por que as métricas são modestas (split honesto vs literatura inflada). → **F-0008, F-0013, F-0017**.
- CNN vs ViT: viés local vence em dados pequenos; ViT data-hungry. → **F-0021**.
- Capacidade: nem pouca (ResNet-18) nem excessiva (ViT-Base) — Swin é ótimo. → **F-0022**.
- Interpretabilidade explica a métrica (ViT polariza porque não localiza). → **F-0021**.
- Limitações: 347/416 pacientes, fatia 2D única, JPG (intensidade perdida), gap só entre 14 papers. → **F-0002, F-0016, F-0015**.

## 7. Conclusão e Trabalhos Futuros
- Síntese das contribuições.
- Trabalhos futuros: 2.5D (**ADR-0011**), busca sistemática, OASIS completo/ADNI, LLRD para ViT (**ADR-0013**), Chefer interpretability.

## 8. Referências
- 14 papers em `docs/Papers/` + fontes externas citadas nos Findings (imbalanced-learn, Nature Sci Reports, etc.).

---

## Apêndices úteis
- FAQ de defesa: `docs/faq-defesa-banca.md` (respostas de banca).
- Reprodutibilidade: ADRs/Findings, splits versionados, seeds, git_commit por run.
