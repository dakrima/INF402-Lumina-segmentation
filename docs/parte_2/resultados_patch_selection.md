# Resultados y pipeline de seleccion inteligente de patches

## 1. Objetivo de la etapa

El objetivo de esta etapa es construir y evaluar una estrategia tecnica de seleccion inteligente de patches para imagenes histopatologicas H&E. Lumina/Histora se mantiene como una herramienta de apoyo para analisis histopatologico; en INF402, la contribucion tecnica principal se concentra en seleccionar patches informativos, diversos y trazables para alimentar una segmentacion semantica posterior.

El flujo principal actual del paper compara `baseline_tiatoolbox` contra `v4_1_medical_embedding_assisted`. `smart_tissue_nuclei_v1`, `smart_tissue_nuclei_v2_light`, `v3_server_quality` y `v4_embedding_assisted` se conservan como iteraciones previas, ablations o soporte interno para trazabilidad/reproducibilidad.

Este trabajo no diagnostica, no calcula RCB, no cuantifica cancer residual como objetivo principal, no reemplaza al patologo y no constituye validacion clinica.

## 2. Resumen del pipeline

El flujo implementado es:

```text
WSI H&E
  -> candidate pool tecnico
  -> baseline_tiatoolbox o v4_1_medical_embedding_assisted
  -> selected patches
  -> metadata CSV/JSON y previews
  -> comparacion tecnica baseline vs selector propuesto
  -> segmentacion semantica posterior como validacion visual/tecnica
```

La idea experimental clave es mantener trazabilidad de pool y el mismo presupuesto de patches por WSI. Los metodos no comparten necesariamente el mismo pool inicial: el baseline usa TIAToolbox/Otsu, mientras que v4.1 usa su propia generacion de candidatos y ranking tecnico. Esta diferencia debe reportarse explicitamente en la comparacion.

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
candidate rows: 1427
selected patches: 16
tissue_mask_method: tiatoolbox_otsu
```

## 4. Selector propuesto: v4_1_medical_embedding_assisted

`v4_1_medical_embedding_assisted` es el selector tecnico propuesto para la comparacion principal. Puede reutilizar funciones internas de `v3_server_quality` como scoring/base tecnica, agrega proxies clasicos de procesamiento de imagen medica y usa UNI como reranker morfologico dentro de candidatos tecnicamente fuertes. Esto no implica que `v3_server_quality` se ejecute como metodo principal.

Incorpora:

- proxies tecnicos de calidad de tincion/intensidad, tejido, textura, nitidez, artefactos y pseudo-celularidad;
- score base reutilizado desde helpers de `v3_server_quality`;
- embeddings UNI como reranking morfologico, no como clasificador clinico;
- seleccion de patches tecnicamente utiles para segmentacion posterior.

En la corrida de cierre:

```text
selector: v4_1_medical_embedding_assisted
candidate rows: 1497
selected patches: 16
segmentation_model_used_for_selection: false
embedding_backend: uni
embedding_model_name: UNI
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

**Candidate pool**: conjunto de candidatos que pasa el filtro inicial de cada metodo. En el flujo principal actual, baseline y v4.1 no comparten necesariamente el mismo pool: baseline usa TIAToolbox/Otsu y v4.1 usa su generacion propia con ranking tecnico.

## 6. Features utilizadas por el selector

`tissue_ratio`: mide que proporcion del patch parece contener tejido y no fondo blanco.

`nuclear_signal` y proxies de pseudo-celularidad: estiman senal asociada a hematoxilina/nucleos. En iteraciones previas se uso `hed_deconvolution`; en v4.1 se integran proxies clasicos de imagen medica y pseudo-celularidad. Son heuristicas basadas en tincion, no segmentacion nuclear ni diagnostico.

`visual_entropy`: mide variabilidad visual. Valores mayores suelen indicar mayor riqueza de intensidades.

`blur_score`: estima nitidez mediante gradientes. Valores mayores sugieren mayor detalle local.

`artifact_penalty`: penaliza patrones tecnicamente poco utiles, como fondo blanco, zonas oscuras, saturacion extrema o baja entropia.

`spatial_penalty`: reduce la preferencia por patches cercanos a otros ya seleccionados.

`feature_diversity_bonus`: favorece candidatos diferentes a los ya seleccionados en el espacio de features normalizadas.

## 7. Estrategia de seleccion

`v4_1_medical_embedding_assisted` opera asi:

1. Carga una WSI con OpenSlide.
2. Genera un thumbnail liviano.
3. Construye una mascara de tejido.
4. Genera su pool propio de candidatos.
5. Scorea una cantidad controlada de candidatos.
6. Calcula proxies tecnicos de calidad, tejido, tincion/textura, nitidez, artefactos y pseudo-celularidad.
7. Normaliza features.
8. Calcula `score_v3_base`, scores medicos tecnicos y reranking morfologico con UNI.
9. Aplica seleccion controlada por utilidad tecnica y diversidad.
10. Selecciona `16` patches.
11. Guarda metadata, imagenes seleccionadas y previews.
12. Compara contra baseline usando metricas tecnicas.

La formula de score no implica diagnostico, RCB ni validacion clinica; solo ordena candidatos por utilidad tecnica esperada.

## 8. Outputs generados

Cada selector genera:

- `selected/`: patches PNG seleccionados;
- `candidate_metadata.csv`: pool auditable de candidatos del metodo;
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
  v4_1_medical_embedding_assisted/
  comparison_baseline_vs_v4_1_medical_embedding_assisted/
```

## 9. Comparacion principal baseline vs v4.1

Resultados principales de auditoria de outputs existentes `baseline_tiatoolbox` vs `v4_1_medical_embedding_assisted`:

| Metrica | Baseline | v4.1 | Lectura tecnica |
| --- | ---: | ---: | --- |
| selected patches | 16 | 16 | Mismo presupuesto. |
| candidate rows | 1427 | 1497 | Los pools son distintos por diseno experimental. |
| pool inicial | TIAToolbox/Otsu | pool propio/manual + ranking tecnico | La diferencia debe reportarse, no ocultarse. |
| selected coordinate overlap | 1 | 1 | Hay una coordenada seleccionada compartida. |
| segmentation_model_used_for_selection | no aplica | false | v4.1 no usa segmentacion para seleccionar. |
| embedding_backend | no aplica | uni | UNI se usa como reranking morfologico, no diagnostico. |

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
- v4.1 tiene mayor costo computacional esperado que el baseline por scoring tecnico y embeddings.

## 11. Como reproducir resultados

Baseline:

```bash
/Users/davidkripper/miniforge3/envs/inf402-lumina-seg/bin/python scripts/06_select_wsi_patches.py \
  --wsi-path "/Users/davidkripper/demoCasesMvpFeria/TCGA-A2-A3XS-01Z-00-DX1.867925C0-91D8-40A0-9FEA-25A635AC31E7.svs" \
  --output-dir "outputs/patch_selection/baseline_tiatoolbox_tcga_a2_a3xs" \
  --selector baseline_tiatoolbox \
  --patch-size 1024 \
  --stride 1024 \
  --max-patches 16 \
  --min-tissue-ratio 0.20 \
  --seed 42 \
  --overwrite
```

Selector propuesto v4.1:

```bash
KMP_DUPLICATE_LIB_OK=TRUE /Users/davidkripper/miniforge3/envs/inf402-lumina-seg/bin/python scripts/06_select_wsi_patches.py \
  --wsi-path "/Users/davidkripper/demoCasesMvpFeria/TCGA-A2-A3XS-01Z-00-DX1.867925C0-91D8-40A0-9FEA-25A635AC31E7.svs" \
  --output-dir "outputs/patch_selection/v4_1_medical_embedding_assisted_tcga_a2_a3xs" \
  --selector v4_1_medical_embedding_assisted \
  --patch-size 1024 \
  --stride 1024 \
  --max-patches 16 \
  --min-tissue-ratio 0.20 \
  --seed 42 \
  --max-candidates-to-score 1000 \
  --feature-size 512 \
  --embedding-model-path /Users/davidkripper/models/uni/pytorch_model.bin \
  --embedding-device cpu \
  --embedding-batch-size 16 \
  --reuse-embedding-cache \
  --cache-embeddings \
  --overwrite
```

Comparacion final:

```bash
conda run -n inf402-lumina-seg python scripts/07_compare_patch_selectors.py \
  --baseline-dir "outputs/patch_selection/baseline_tiatoolbox_tcga_a2_a3xs" \
  --smart-dir "outputs/patch_selection/v4_1_medical_embedding_assisted_tcga_a2_a3xs" \
  --output-dir "outputs/patch_selection/comparison_baseline_vs_v4_1_medical_embedding_assisted" \
  --feature-size 256 \
  --overwrite
```
