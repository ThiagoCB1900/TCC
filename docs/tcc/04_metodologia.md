# 4. Metodologia

> Rascunho em texto corrido (português acadêmico). Marcadores `[ref: ...]` indicam a fonte interna (ADR/Finding) — remover na versão final, convertendo em citação/figura quando aplicável.

## 4.1 Conjunto de dados

Utilizou-se a versão pré-processada do **OASIS-1** (*Open Access Series of Imaging Studies*, fase *cross-sectional*) disponibilizada publicamente na plataforma Kaggle (`ninadaithal/imagesoasis`), composta por fatias axiais bidimensionais em formato JPEG, extraídas de imagens de ressonância magnética ponderada em T1 e já submetidas a remoção de crânio (*skull stripping*). Cada arquivo segue a nomenclatura `OAS1_XXXX_MR{N}_mpr-{N}_{NNN}.jpg`, na qual `XXXX` identifica o sujeito, `MR{N}` a sessão de aquisição, `mpr-{N}` a N-ésima aquisição MPRAGE (réplicas para redução de ruído) e `{NNN}` o índice da fatia axial (faixa 100–160) [ref: F-0001].

A análise exploratória (Seção 5.1) contabilizou **86.437 fatias de 347 pacientes** distribuídos em quatro rótulos clínicos baseados na *Clinical Dementia Rating* (CDR). Observou-se forte desbalanceamento: a classe *Non Demented* concentra 77,8% das fatias, enquanto *Moderate Dementia* possui apenas **2 pacientes** [ref: F-0002]. Cabe registrar que o número de pacientes (347) difere tanto do anunciado pela plataforma (461) quanto do OASIS-1 oficial (416 sujeitos); a versão Kaggle corresponde a ~83% do conjunto original, provavelmente por exclusão de sujeitos sem CDR válido — limitação assumida e discutida na Seção 6 [ref: F-0002].

Verificou-se, ainda, que cada imagem é uma **única fatia axial** (496×248 pixels) com os três canais RGB idênticos (escala de cinza promovida na conversão), e não uma composição de vistas — hipótese inicial descartada após inspeção visual sistemática [ref: F-0003].

## 4.2 Esquema de classes

Em razão de *Moderate Dementia* conter apenas dois pacientes — o que inviabiliza um particionamento por sujeito com representação dessa classe nos três conjuntos —, adotou-se um **esquema de três classes**: *Non Demented* (266 pacientes), *Very Mild Dementia* (58) e *Mild or Moderate Dementia* (23, resultante da fusão de *Mild* e *Moderate*). A fusão preserva a granularidade clínica na fronteira mais relevante (pré-clínico vs. clínico) e agrupa estágios já diagnosticados [ref: ADR-0001].

## 4.3 Particionamento por paciente

Cada paciente contribui, em média, com cerca de 244 fatias. Um particionamento aleatório por fatia colocaria o mesmo indivíduo simultaneamente em treino e teste, levando o modelo a memorizar características anatômicas individuais em vez de aprender marcadores da doença — inflando artificialmente as métricas. Por isso, adotou-se **particionamento estratificado por identificador de paciente** (*subject ID*), na proporção 70/15/15 (treino/validação/teste), com semente fixa, estratificado pela classe [ref: ADR-0002]. O procedimento, implementado em `src/data/splits.py`, garante conjuntos de sujeitos disjuntos (verificação automática) e proporções por classe dentro de ~1% da distribuição global. O particionamento resultante (`split_v1`) contém 242 pacientes em treino, 52 em validação e 53 em teste, e é versionado para reprodutibilidade entre máquinas.

Essa decisão é o principal diferencial metodológico do trabalho frente a parte da literatura no mesmo conjunto de dados, que utiliza particionamento por fatia (Seção 3) [ref: F-0008, F-0014].

## 4.4 Pré-processamento e *data augmentation*

As fatias foram redimensionadas para 224×224 pixels por *squash* (deformação direta para o formato quadrado) e normalizadas com as estatísticas do ImageNet, mantendo os três canais para compatibilidade com os pesos pré-treinados [ref: ADR-0004]. Os métodos de normalização específicos de MRI (Nyúl, WhiteStripe) não se aplicam, pois a escala de intensidade original do exame foi perdida na conversão para JPEG [ref: F-0016].

O *data augmentation* foi aplicado **exclusivamente ao conjunto de treino**, após o particionamento, evitando vazamento de informação: espelhamento horizontal, rotação de ±15°, translação de até 5% e variação de brilho/contraste [ref: ADR-0006, ADR-0010]. Conjuntos de validação e teste permaneceram intactos. Essa separação — *augmentation* e balanceamento apenas no treino — é uma regra metodológica fundamental: aplicá-los à validação/teste introduz viés e distorce as métricas [ref: ADR-0011, F-0016].

## 4.5 Tratamento do desbalanceamento

O desbalanceamento foi tratado por **função de perda ponderada por classe** (*weighted cross-entropy*), com pesos calculados pela fórmula "balanced" sobre o conjunto de treino (peso inversamente proporcional à frequência da classe) [ref: ADR-0007]. Optou-se por essa abordagem (sensível ao custo) em vez de reamostragem (*oversampling*/SMOTE) porque o número reduzido de pacientes na classe minoritária (17 em treino) tornaria a duplicação de fatias uma diversidade ilusória. Reforça-se que o balanceamento atua **apenas no treino**; no teste, o desbalanceamento é tratado pelas métricas, não pelos dados [ref: F-0016].

## 4.6 Arquiteturas avaliadas

Compararam-se quatro arquiteturas, todas com pesos pré-treinados no ImageNet (via biblioteca `timm`) e *fine-tuning* completo [ref: ADR-0012, ADR-0008]:

- **ResNet-50** (~23,5M parâmetros) — linha de base convolucional;
- **ResNet-18** (~11,2M) — ablação de capacidade;
- **ViT-Base/16** (~85,8M) — *transformer* puro;
- **Swin-T** (~27,5M) — *transformer* hierárquico com atenção em janelas.

Todas recebem entrada 224×224 e têm a camada de classificação reinicializada para três saídas.

## 4.7 Protocolo de treino

O treino foi implementado em PyTorch puro (sem *frameworks* de alto nível, favorecendo a auditabilidade) [ref: ADR-0008]. Utilizou-se o otimizador AdamW com **taxa de aprendizado diferenciada** — menor no *backbone* pré-treinado e maior na cabeça de classificação reinicializada —, *weight decay*, *cosine annealing* com *warmup*, *gradient clipping* e parada antecipada (*early stopping*) monitorando a *balanced accuracy* de validação [ref: ADR-0010]. Os hiperparâmetros foram refinados em iterações sucessivas para conter o sobreajuste: a configuração V2 estabilizou a ResNet [ref: ADR-0010] e a V3 — com regularização mais forte (taxa do *backbone* 5×10⁻⁶, *stochastic depth* 0,2, *weight decay* 0,2, *warmup* de 3 épocas) — foi necessária para os *transformers* [ref: ADR-0013].

O treino dos modelos foi executado em **Kaggle Notebooks** (GPU T4/P100 gratuita), ambiente escolhido por hospedar o conjunto de dados nativamente em disco local, eliminando o gargalo de E/S que inviabilizou tentativas anteriores em outro ambiente [ref: ADR-0009, F-0011, F-0012]. O desenvolvimento, a análise exploratória e a interpretabilidade rodaram localmente em CPU.

## 4.8 Métricas e comparação estatística

Dado o desbalanceamento, as métricas primárias são **balanced accuracy, macro-F1 e AUC macro** (um-contra-resto), reportando-se também F1 por classe e matriz de confusão; a acurácia bruta é apresentada apenas como referência secundária, por ser enganosa neste cenário [ref: ADR-0005]. A comparação par-a-par entre arquiteturas usou o **teste de McNemar** sobre as predições do mesmo conjunto de teste. Ressalta-se que o McNemar avalia acertos brutos e, portanto, deve ser lido em conjunto com a *balanced accuracy* — sob risco de conclusão invertida em dados desbalanceados [ref: F-0021].

## 4.9 Interpretabilidade

Para explicar as predições, empregou-se o método nativo de cada família: **Attention Rollout** (Abnar & Zuidema, 2020) para o ViT-Base, que agrega as matrizes de atenção das camadas; e **Grad-CAM** (Selvaraju et al., 2017) para a ResNet e o Swin-T, este com reorganização dos *tokens* em grade espacial [ref: ADR-0014]. Os mapas foram sobrepostos às fatias originais e analisados quanto à plausibilidade clínica das regiões destacadas (Seção 5.6).
