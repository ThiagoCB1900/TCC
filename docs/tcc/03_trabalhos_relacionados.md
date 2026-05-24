# 3. Trabalhos Relacionados

> Rascunho em texto corrido. Marcadores `[ref: ...]` indicam a fonte interna — remover na versão final, substituindo por citações ABNT (\cite{...} no ueceTeX2). As métricas e afirmações de cada trabalho foram auditadas a partir dos PDFs (Seção de método de cada um).

## 3.1 Transformers em neuroimagem de Alzheimer

A aplicação de *Vision Transformers* (ViT) e suas variantes à classificação de Alzheimer a partir de ressonância magnética tem crescido desde 2021. Uma revisão sistemática recente, conduzida com a ferramenta QUADAS-2, aponta porém um problema transversal: cerca de 31% dos estudos analisados não detalham adequadamente a seleção de pacientes, resultando em alto risco de viés [ref: F-0014, detection_of_Alzheimer]. Esse achado motiva o eixo central deste capítulo: **como cada trabalho relacionado particiona seus dados** — decisão que, como se argumenta a seguir, determina a confiabilidade das métricas reportadas.

## 3.2 A questão do particionamento dos dados

Em conjuntos de neuroimagem, cada paciente contribui com múltiplas imagens (fatias de um volume, ou múltiplas sessões). Quando o particionamento treino/teste é feito por imagem (ou fatia) de forma aleatória, imagens de um mesmo paciente acabam em ambos os conjuntos, permitindo que o modelo memorize características individuais e infle artificialmente as métricas — um vazamento de dados (*data leakage*). O particionamento correto agrupa por paciente (*subject-level split*).

A auditoria dos trabalhos relacionados [ref: F-0014] revela um quadro nuançado, sintetizado na Tabela 3.1:

**Tabela 3.1 — Particionamento de dados nos trabalhos relacionados auditados**

| Trabalho | Dataset | Particionamento | Por paciente? |
|---|---|---|---|
| RanCom-ViT (Lu et al., 2025) | OASIS-1 (Kaggle) | aleatório por fatia | Não |
| Joint Transformer (2024) | ADNI 3D | aleatório por volume | Parcial¹ |
| Leveraging Swin | ADNI (dMRI) | por sujeito (15% hold-out + 5-fold) | Sim |
| VGG-TSwinformer | ADNI | por sujeito | Sim |
| Vision-ViT-CNN | ADNI | por sujeito | Sim |
| Notebook CNN (comparativo) | uraninjo | por imagem | Impossível² |

¹ Particiona por volume 3D, mas não controla vazamento longitudinal entre sessões do mesmo paciente no ADNI.
² O conjunto não preserva identificadores de paciente (Seção 3.5).

Observa-se que os trabalhos metodologicamente mais rigorosos **adotam particionamento por paciente** e, em um caso, criticam explicitamente a prática contrária: os autores de *Leveraging Swin* afirmam que "*evaluation design can inflate headline metrics*" sob particionamentos aleatórios [ref: F-0014]. A divisão por paciente, portanto, não é uma exigência arbitrária deste trabalho, mas a prática consolidada entre os estudos de referência.

## 3.3 RanCom-ViT: o contraponto metodológico no mesmo conjunto de dados

O trabalho mais diretamente comparável é o RanCom-ViT (Lu, Zhang & Yao, 2025), por utilizar **exatamente o mesmo conjunto de dados** (a versão Kaggle do OASIS-1) e a mesma família de arquitetura (ViT). Os autores reportam acurácia de 99,54% [ref: F-0008]. Contudo, a auditoria de sua metodologia revela três limitações: (i) o particionamento é aleatório por fatia, estratificado por classe e não por paciente — *"we split the dataset randomly into 80% for training and 20% for testing... we stratified the dataset by class"*; (ii) a descrição do plano de corte é incorreta, referindo-se às fatias como "sagitais" quando são axiais; e (iii) as métricas não incluem *balanced accuracy*, apesar do forte desbalanceamento [ref: F-0008].

Esse trabalho constitui, assim, o **oponente metodológico** desta monografia: não um alvo a ser superado em valor numérico — comparação que seria inválida —, mas um contraponto que evidencia como o particionamento por fatia produz métricas otimistas. A acurácia de 99,54% reflete, em grande medida, a capacidade do modelo de reconhecer pacientes vistos no treino, não de generalizar para novos pacientes.

## 3.4 Trabalhos com particionamento por paciente (ADNI)

Três trabalhos auditados adotam particionamento por paciente, todos sobre o conjunto ADNI [ref: F-0014, F-0015]:

- **Leveraging Swin** (Swin Transformer + LoRA sobre dMRI): *balanced accuracy* de 95,2% na tarefa binária AD vs. CN. Além do rigor no particionamento (15% de sujeitos em *hold-out* + validação cruzada de 5 *folds* agrupada por sujeito), o trabalho disponibiliza os identificadores de sujeito utilizados.
- **Vision-ViT-CNN** (ViT + CNN sobre ADNI 3D): acurácia de 92,14% na tarefa de três classes (NC/MCI/AD), com *"subject-level split strategy"* explícita e tratamento de desbalanceamento por perda ponderada — abordagem idêntica à adotada nesta monografia.
- **VGG-TSwinformer** (VGG-16 + Swin temporal, dados longitudinais): acurácia de 77,2% na tarefa pMCI vs. sMCI, justificando explicitamente que amostras de um mesmo sujeito não devem cruzar os conjuntos.

A diferença de magnitude entre esses valores e os do RanCom-ViT ilustra o efeito do rigor: tarefas mais difíceis, com particionamento honesto, reportam métricas substancialmente menores.

## 3.5 Conjuntos sem identificação de paciente

Como contraponto adicional, analisou-se o conjunto `uraninjo/augmented-alzheimer-mri-dataset`, popular no Kaggle e utilizado no notebook de CNN tomado como comparativo informal [ref: F-0005]. A auditoria demonstrou que, nesse conjunto, **o particionamento por paciente é impossível**: a nomenclatura dos arquivos não contém identificador de sujeito, os metadados EXIF estão vazios e o mesmo prefixo numérico aparece em classes distintas — o que seria clinicamente impossível se representasse um paciente [ref: F-0014]. A informação de paciente foi destruída na curadoria, de modo que qualquer trabalho que utilize esse conjunto não pode garantir — nem auditar — a ausência de vazamento.

## 3.6 Posicionamento e lacuna

O mapeamento dos trabalhos auditados segundo três eixos — conjunto de dados, arquitetura e tipo de particionamento — revela que a interseção **ViT + OASIS-1 + particionamento por paciente está vazia** [ref: F-0015]: os trabalhos que adotam particionamento por paciente migraram para o ADNI, enquanto os que permanecem no OASIS-1 (2D) utilizam particionamento por fatia. É nessa lacuna que esta monografia se posiciona, aplicando ViT, Swin e uma linha de base convolucional ao OASIS-1 com particionamento rigoroso por paciente, métricas balanceadas e análise de interpretabilidade.

> Nota: a confirmação definitiva da lacuna requer busca sistemática (Google Scholar, PubMed, IEEE Xplore); o presente mapeamento abrange os trabalhos reunidos no projeto. Recomenda-se a busca antes da versão final [ref: F-0015].
