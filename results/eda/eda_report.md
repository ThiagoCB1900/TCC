# EDA — OASIS-1 (versão Kaggle pré-processada)

- Total de slices: **86.437**
- Total de pacientes únicos: **347**
- Aquisições únicas (subject × session × mpr): **1417**
- Slice indices observados: 100–160
- mpr observado: [1, 2, 3, 4, 5, 6]
- Sessões observadas: MR [1, 2]

## Sanity checks
- ✅ Nenhum sujeito aparece em mais de uma classe.


## Distribuição (4 classes — original do Kaggle)

| class_4      |   n_subjects |   n_acquisitions |   n_slices |   pct_slices |   slices_per_subject_mean |
|:-------------|-------------:|-----------------:|-----------:|-------------:|--------------------------:|
| non_demented |          266 |             1102 |      67222 |        77.77 |                     252.7 |
| very_mild    |           58 |              225 |      13725 |        15.88 |                     236.6 |
| mild         |           21 |               82 |       5002 |         5.79 |                     238.2 |
| moderate     |            2 |                8 |        488 |         0.56 |                     244   |

## Distribuição (3 classes — Mild+Moderate fundidos)

| class_3          |   n_subjects |   n_acquisitions |   n_slices |   pct_slices |   slices_per_subject_mean |
|:-----------------|-------------:|-----------------:|-----------:|-------------:|--------------------------:|
| non_demented     |          266 |             1102 |      67222 |        77.77 |                     252.7 |
| very_mild        |           58 |              225 |      13725 |        15.88 |                     236.6 |
| mild_or_moderate |           23 |               90 |       5490 |         6.35 |                     238.7 |

## Distribuição (binário — fallback do plano original)

| class_binary   |   n_subjects |   n_acquisitions |   n_slices |   pct_slices |   slices_per_subject_mean |
|:---------------|-------------:|-----------------:|-----------:|-------------:|--------------------------:|
| non_demented   |          266 |             1102 |      67222 |        77.77 |                     252.7 |
| demented       |           81 |              315 |      19215 |        22.23 |                     237.2 |

## Estatísticas das imagens (amostra)

| class_4      |   width |   height | mode   |   intensity_mean |   intensity_std |   fraction_zero |
|:-------------|--------:|---------:|:-------|-----------------:|----------------:|----------------:|
| mild         |     496 |      248 | RGB    |            39.59 |           44.93 |            0.3  |
| moderate     |     496 |      248 | RGB    |            40.57 |           45.25 |            0.3  |
| non_demented |     496 |      248 | RGB    |            42.98 |           46.12 |            0.31 |
| very_mild    |     496 |      248 | RGB    |            42.1  |           44.17 |            0.28 |

## Figuras geradas
- `figures/class_distribution.png`
- `figures/class_distribution_3_and_binary.png`
- `figures/examples_grid.png`
- `figures/intensity_mean_by_class.png`
- `figures/slice_idx_distribution.png`
- `figures/slices_per_subject.png`