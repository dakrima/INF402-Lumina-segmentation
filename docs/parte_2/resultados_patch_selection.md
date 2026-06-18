# Resultados y pipeline de seleccion inteligente de patches

## 1. Objetivo de la etapa

El objetivo de esta etapa es construir y evaluar una estrategia tecnica de seleccion inteligente de patches para imagenes histopatologicas H&E. Lumina/Histora se mantiene como una herramienta de apoyo para analisis histopatologico; en INF402, la contribucion tecnica principal se concentra en seleccionar patches informativos, diversos y trazables para alimentar una segmentacion semantica posterior.

El selector candidato final congelado por ahora es `smart_tissue_nuclei_v2_light`. El baseline comparativo es `baseline_tiatoolbox` y `smart_tissue_nuclei_v1` se conserva como version intermedia/ablation.

Este trabajo no diagnostica, no calcula RCB, no cuantifica cancer residual como objetivo principal, no reemplaza al patologo y no constituye validacion clinica.

## 2. Resumen del pipeline

El flujo implementado es:

```text
WSI H&E
  -> thumbnail liviano
  -> mascara de tejido
  -> candidate pool comun thumbnail-filtered
  -> baseline_tiatoolbox o smart_tissue_nuclei_v2_light
  -> selected patches
  -> metadata CSV/JSON y previews
  -> comparacion tecnica baseline vs selector propio
  -> segmentacion semantica posterior como validacion visual/tecnica
```

La idea experimental clave es mantener trazabilidad de pool y el mismo presupuesto de patches por WSI. En corridas previas, ambos metodos seleccionaban `16` patches desde un pool comun de `1497` candidatos filtrados por thumbnail; con el baseline real TIAToolbox/Otsu, cualquier diferencia de pool debe quedar reportada explicitamente en la comparacion.

## 3. Baseline: baseline_tiatoolbox

`baseline_tiatoolbox` es el punto de referencia. Usa TIAToolbox `SlidingWindowPatchExtractor` sobre la WSI con mascara automatica Otsu, `min_mask_ratio` y seleccion reproducible con `seed`.

Sus propiedades principales son:

- usa ventana deslizante TIAToolbox, mascara Otsu y `min_tissue_ratio`;
- mantiene `candidate_metadata.csv` como pool auditable de candidatos;
- guarda solo los patches seleccionados en `selected/`;
- no calcula ranking inteligente, senal nuclear, HED, embeddings, cuotas espaciales ni diversidad por features.

En la corrida de cierre:

```text
selector: baseline_tiatoolbox
candidate rows: 1497
selected patches: 16
runtime_seconds: 2.571
```

## 4. Selector propio final: smart_tissue_nuclei_v2_light

`smart_tissue_nuclei_v2_light` es el selector propio candidato final de esta etapa. Mantiene el mismo pool comun de candidatos, pero scorea una muestra controlada de candidatos y prioriza patches por calidad tecnica, contenido de tejido, proxy nuclear/hematoxilina, bajo ruido y diversidad.

Incorpora tres extensiones sobre v1:

- HED color deconvolution como proxy de senal hematoxilina/nuclear;
- spatial quotas suaves para evitar concentrar todos los patches en una sola region;
- feature diversity simple para favorecer patches con caracteristicas distintas.

En la corrida de cierre:

```text
selector: smart_tissue_nuclei_v2_light
nuclear_proxy: hed_deconvolution
spatial_strategy: quotas
diversity_strategy: farthest_feature
candidate rows: 1497
candidates scored: 500
selected patches: 16
regions_covered: 12
active_regions: 13
quota_fill_rate: 0.9231
runtime_seconds: 22.185
```

## 5. Explicacion de conceptos clave

**WSI**: una Whole Slide Image es una imagen digital de una lamina completa. Puede tener dimensiones gigapixel, por lo que no se procesa completa como una imagen comun.

**Patch**: recorte pequeno de la WSI. En esta etapa se usan patches de `1024x1024` pixeles para reducir costo computacional y permitir evaluacion local.

**H&E**: tincion comun en histopatologia. Hematoxilina tiñe principalmente estructuras nucleares en tonos azul/morado, mientras que eosina tiñe citoplasma y matriz extracelular en tonos rosados.

**HED**: espacio de color usado para separar aproximadamente componentes de tincion, incluyendo hematoxilina y eosina.

**Color deconvolution**: tecnica que estima contribuciones de tinciones a partir de color RGB. En este proyecto se usa como heuristica, no como verdad biologica exacta.

**Hematoxilina**: componente de H&E asociado principalmente a estructuras nucleares. Una mayor senal hematoxilina puede ser util como proxy tecnico de contenido celular/nuclear.

**Eosina**: componente de H&E asociado a citoplasma, matriz extracelular y otros tejidos en tonos rosados.

**Mascara de tejido**: mascara simple que separa regiones con tejido de fondo blanco o vacio. Sirve para construir el pool inicial de candidatos.

**Entropia visual**: medida de variabilidad de intensidades. Ayuda a evitar patches demasiado uniformes o con poca informacion visual.

**Blur score / nitidez**: proxy de detalle local basado en variacion de gradientes. Patches con mas textura fina tienden a tener mayor valor.

**Artifact penalty**: penalizacion por caracteristicas no deseadas, como exceso de fondo blanco, regiones muy oscuras, saturacion extrema o baja informacion.

**Spatial quotas**: reparto suave de seleccion entre regiones de la WSI. No obliga a elegir patches malos; limita concentracion excesiva.

**Feature diversity**: bonificacion para seleccionar patches distintos entre si en el espacio de features, reduciendo redundancia.

**Candidate pool**: conjunto comun de candidatos que paso el filtro thumbnail. Baseline y selector propio compiten sobre ese mismo pool.

## 6. Features utilizadas por el selector

`tissue_ratio`: mide que proporcion del patch parece contener tejido y no fondo blanco.

`nuclear_signal`: estima senal asociada a hematoxilina/nucleos. En `smart_tissue_nuclei_v2_light` se usa `hed_deconvolution`. Esto es un proxy heuristico basado en tincion, no segmentacion nuclear.

`visual_entropy`: mide variabilidad visual. Valores mayores suelen indicar mayor riqueza de intensidades.

`blur_score`: estima nitidez mediante gradientes. Valores mayores sugieren mayor detalle local.

`artifact_penalty`: penaliza patrones tecnicamente poco utiles, como fondo blanco, zonas oscuras, saturacion extrema o baja entropia.

`spatial_penalty`: reduce la preferencia por patches cercanos a otros ya seleccionados.

`feature_diversity_bonus`: favorece candidatos diferentes a los ya seleccionados en el espacio de features normalizadas.

## 7. Estrategia de seleccion

`smart_tissue_nuclei_v2_light` opera asi:

1. Carga una WSI con OpenSlide.
2. Genera un thumbnail liviano.
3. Construye una mascara de tejido.
4. Genera un pool comun de candidatos.
5. Scorea una cantidad controlada de candidatos.
6. Calcula features de tejido, nucleo/tincion, entropia, nitidez y artefactos.
7. Normaliza features.
8. Calcula `score_raw`.
9. Aplica cuotas espaciales suaves y diversidad por features.
10. Selecciona `16` patches.
11. Guarda metadata, imagenes seleccionadas y previews.
12. Compara contra baseline usando metricas tecnicas.

La formula de score no se cambio en el cierre; esta etapa congela el selector candidato final y ordena resultados/documentacion.

## 8. Outputs generados

Cada selector genera:

- `selected/`: patches PNG seleccionados;
- `candidate_metadata.csv`: pool comun de candidatos thumbnail-filtered;
- `selected_metadata.csv`: solo patches seleccionados;
- `selection_summary.json`: resumen de corrida;
- `method_config.json`: configuracion del metodo;
- `patch_selection_preview.png`: preview visual.

La comparacion genera:

- `comparison_summary.json`;
- `comparison_metrics.csv`;
- `comparison_selected_patches.csv`;
- `selected_overlap.csv`;
- `comparison_preview.png`;
- `comparison_preview_selected_only.png`;
- `comparison_notes.md`.

El snapshot limpio queda en:

```text
outputs/final_patch_selection/
  baseline_tiatoolbox/
  smart_tissue_nuclei_v2_light/
  comparison_baseline_vs_smart_v2_light/
```

## 9. Comparacion final baseline vs smart_v2_light

Resultados principales de la corrida `baseline_tiatoolbox` vs `smart_tissue_nuclei_v2_light`:

| Metrica | Baseline | smart_v2_light | Lectura tecnica |
| --- | ---: | ---: | --- |
| selected patches | 16 | 16 | Mismo presupuesto. |
| tissue_ratio promedio | 0.8445 | 0.8924 | v2 selecciona patches con mas tejido promedio. |
| nuclear_signal_rgb promedio | 0.0917 | 0.0946 | Diferencia leve en proxy RGB legacy. |
| nuclear_signal_hed promedio | 0.2715 | 0.4507 | v2 prioriza mayor senal HED/hematoxilina. |
| visual_entropy promedio | 0.8171 | 0.8613 | v2 aumenta variabilidad visual promedio. |
| blur_score promedio | 0.0111 | 0.0177 | v2 prioriza patches con mayor nitidez estimada. |
| artifact_penalty promedio | 0.0501 | 0.0330 | v2 reduce penalizacion por artefactos. |
| overlap | 0 | 0 | Selecciona un conjunto distinto al baseline. |
| jaccard | 0.0000 | 0.0000 | No hubo coincidencia entre seleccionados. |
| mean_pairwise_distance | 24790.42 | 26443.62 | v2 mejora distancia media entre pares. |
| spatial_coverage_approx | 0.5061 | 0.4930 | v2 queda levemente menor en cobertura bbox aproximada. |
| runtime_seconds | 2.571 | 22.185 | v2 es mas costoso por scoring de features. |
| regions_covered | 0/no aplica | 12 | v2 cubre 12 de 13 regiones activas. |
| quota_fill_rate | no aplica | 0.9231 | Cuotas suaves cubren la mayoria de regiones activas. |

La comparacion es tecnica. Una mejora en estas metricas no implica diagnostico, superioridad clinica ni desempeño de segmentacion garantizado. Dice/IoU deberian evaluarse aparte cuando exista ground truth compatible.

## 10. Limitaciones

- Los resultados son tecnicos y heuristicos.
- No implican diagnostico.
- No calculan RCB.
- No reemplazan revision patologica.
- HED es un proxy de tincion, no segmentacion nuclear.
- La comparacion actual depende de los casos evaluados.
- La generalizacion requiere mas WSIs y/o evaluacion con ground truth.
- Las metricas Dice, IoU o pixel accuracy corresponden a segmentacion posterior, no a la seleccion de patches por si sola.
- `smart_tissue_nuclei_v2_light` tiene mayor costo computacional que el baseline.

## 11. Como reproducir resultados

Baseline:

```bash
conda run -n inf402-lumina-seg python scripts/06_select_wsi_patches.py \
  --wsi-path "/Users/davidkripper/demoCasesMvpFeria/TCGA-A2-A3XS-01Z-00-DX1.867925C0-91D8-40A0-9FEA-25A635AC31E7.svs" \
  --output-dir "outputs/patch_selection/baseline_tcga_a2_a3xs" \
  --selector baseline_tiatoolbox \
  --patch-size 1024 \
  --stride 1024 \
  --max-patches 16 \
  --min-tissue-ratio 0.20 \
  --seed 42 \
  --overwrite
```

Selector candidato final:

```bash
conda run -n inf402-lumina-seg python scripts/06_select_wsi_patches.py \
  --wsi-path "/Users/davidkripper/demoCasesMvpFeria/TCGA-A2-A3XS-01Z-00-DX1.867925C0-91D8-40A0-9FEA-25A635AC31E7.svs" \
  --output-dir "outputs/patch_selection/smart_v2_light_tcga_a2_a3xs" \
  --selector smart_tissue_nuclei_v2_light \
  --patch-size 1024 \
  --stride 1024 \
  --max-patches 16 \
  --min-tissue-ratio 0.20 \
  --seed 42 \
  --max-candidates-to-score 500 \
  --feature-size 256 \
  --lambda-spatial 0.15 \
  --nuclear-proxy hed_deconvolution \
  --spatial-strategy quotas \
  --quota-grid 4x4 \
  --quota-min-score-quantile 0.25 \
  --diversity-strategy farthest_feature \
  --feature-diversity-weight 0.10 \
  --overwrite
```

Comparacion final:

```bash
conda run -n inf402-lumina-seg python scripts/07_compare_patch_selectors.py \
  --baseline-dir "outputs/patch_selection/baseline_tcga_a2_a3xs" \
  --smart-dir "outputs/patch_selection/smart_v2_light_tcga_a2_a3xs" \
  --output-dir "outputs/patch_selection/comparison_baseline_vs_smart_v2_light_tcga_a2_a3xs" \
  --feature-size 256 \
  --overwrite
```
