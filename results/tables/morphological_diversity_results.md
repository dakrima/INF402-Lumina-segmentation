# Diversidad morfológica aproximada en el espacio UNI

Se analizaron los 16 patches previamente seleccionados por método en cada una de las nueve WSI. Todos los embeddings se reutilizaron desde los cachés UNI originales y fueron normalizados por su norma L2 antes de calcular las distancias coseno.

## Comparación agregada

| Métrica | Baseline: media ± DE | v4.1: media ± DE | Baseline: mediana [Q1, Q3] | v4.1: mediana [Q1, Q3] | Δ pareada media | Δ pareada mediana | v4.1 / baseline / empates |
|---|---:|---:|---:|---:|---:|---:|---:|
| Distancia coseno media entre pares | 0.591883 ± 0.057795 | 0.551488 ± 0.057035 | 0.598437 [0.572645, 0.633393] | 0.560458 [0.530769, 0.575548] | -0.040395 | -0.037979 | 2 / 7 / 0 |
| Distancia coseno media al vecino más cercano | 0.386813 ± 0.052963 | 0.319636 ± 0.055559 | 0.386770 [0.363481, 0.434020] | 0.328628 [0.293703, 0.354816] | -0.067177 | -0.047129 | 1 / 8 / 0 |

Δ corresponde a v4.1 menos baseline. DE: desviación estándar; Q1 y Q3: cuartiles 25 y 75.

## Diferencias pareadas por WSI

| WSI | Δ distancia media entre pares | Δ distancia media al vecino más cercano |
|---|---:|---:|
| TCGA-A2-A04T | -0.144282 | -0.115927 |
| TCGA-A2-A0CM | -0.001987 | -0.039890 |
| TCGA-A2-A3XS | +0.121196 | +0.027980 |
| TCGA-A7-A0DA | -0.057846 | -0.047129 |
| TCGA-AO-A03U | -0.147678 | -0.159262 |
| TCGA-C8-A26Y | +0.020534 | -0.030776 |
| TCGA-E2-A1L7 | -0.099754 | -0.133066 |
| TCGA-GM-A3XL | -0.015757 | -0.074570 |
| TCGA-OL-A66I | -0.037979 | -0.031954 |

## Interpretación descriptiva

En la distancia coseno media entre pares, v4.1 obtuvo un valor mayor en 2 de 9 WSI, mientras que el baseline fue mayor en 7 y se observaron 0 empates. La diferencia pareada media fue -0.040395.

Para la distancia media al vecino morfológico más cercano, v4.1 fue mayor en 1 de 9 WSI, el baseline fue mayor en 8 y se observaron 0 empates. La diferencia pareada media fue -0.067177.

## Propuesta breve para la sección III

La diversidad morfológica aproximada se evaluó mediante la distancia coseno entre embeddings UNI de los patches seleccionados. La distancia media entre pares fue 0.591883 ± 0.057795 para el baseline y 0.551488 ± 0.057035 para v4.1, con una diferencia pareada media de -0.040395. La distancia media al vecino morfológico más cercano fue 0.386813 ± 0.052963 y 0.319636 ± 0.055559, respectivamente, con una diferencia pareada media de -0.067177. Estos resultados son descriptivos y corresponden exclusivamente a la diversidad aproximada en el espacio de representaciones de UNI.
