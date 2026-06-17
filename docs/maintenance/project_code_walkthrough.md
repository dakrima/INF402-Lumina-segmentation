# Walkthrough técnico del proyecto Lumina/Histora

Fecha de referencia: 2026-06-16  
Repositorio: `/Users/davidkripper/INF402-Lumina-segmentation`

## 1. Resumen general del proyecto

Este repositorio contiene el pipeline técnico de INF402 para Lumina/Histora. Su
objetivo es apoyar el análisis visual de imágenes histopatológicas H&E mediante
selección técnica de patches, segmentación semántica posterior y comparación
técnica de resultados.

El foco técnico actual no es diagnosticar ni medir cáncer residual. El foco es
seleccionar regiones informativas y trazables desde WSI para alimentar una etapa
posterior de segmentación. La segmentación genera máscaras y overlays
revisables, pero esas salidas son predicciones técnicas del modelo, no ground
truth clínico.

El proyecto se organiza en tres grandes bloques:

- Selección de patches: genera candidatos desde WSI y selecciona un subconjunto
  bajo un presupuesto fijo.
- Segmentación técnica: aplica un modelo preentrenado de TIAToolbox sobre los
  patches seleccionados.
- Comparación técnica: compara selectores y outputs de segmentación sin afirmar
  desempeño clínico.

Diagrama de flujo:

```text
WSI H&E
  -> generación de candidatos
  -> selección baseline/smart
  -> comparación de selectores
  -> segmentación sobre patches seleccionados
  -> comparación técnica de segmentación
  -> documentación/análisis posterior
```

Lo que el proyecto no hace:

- no diagnostica;
- no calcula RCB;
- no reemplaza al patólogo;
- no valida clínicamente resultados;
- no interpreta una máscara como verdad clínica;
- no sube WSI, checkpoints ni outputs pesados al repo.

## 2. Estructura general del repositorio

### `archive/`

Carpeta para código histórico que ya no forma parte de la ruta principal. Debe
versionarse solo si contiene referencias pequeñas y útiles para entender la
evolución del proyecto.

### `archive/legacy_small_image_pipeline/`

Archivo histórico del flujo inicial para imágenes pequeñas. Contiene:

- `scripts/03_extract_patches.py`
- `src/patching/`
- `src/config/`
- `notebooks/00_exploration.ipynb`

Ese flujo trabajaba sobre imágenes pequeñas comunes y no sobre WSI como ruta
principal. Se conserva como referencia. Si alguien quiere ejecutarlo desde
`archive/`, probablemente deberá ajustar imports y paths.

### `data/`

Estructura local para datos. Incluye `raw/`, `external/`, `interim/` y
`processed/`. El contenido real está ignorado por Git; solo se versionan
`.gitkeep`. No se deben subir WSI, datasets clínicos ni datos sensibles.

### `docs/`

Documentación del proyecto. Se divide en documentación de etapa, mantenimiento y
auditoría.

### `docs/maintenance/`

Documentación interna de mantenimiento. Actualmente contiene:

- `repo_cleanup_audit.md`: auditoría de ordenamiento del repo.
- `project_code_walkthrough.md`: este walkthrough técnico.

### `docs/parte_1/`

Carpeta reservada para documentación de Parte I. Según la estructura actual,
solo contiene `.gitkeep`; los documentos principales de Parte I pueden vivir
fuera del repo, por ejemplo en OneDrive.

### `docs/parte_2/`

Documentación técnica de implementación y resultados. Contiene:

- `plan_codigo.md`
- `resultados_patch_selection.md`
- `auditoria_cierre_patch_selection.md`

### `outputs/`

Carpeta local para artefactos generados. Su contenido real no debe versionarse.
Las subcarpetas principales son:

- `outputs/patch_selection/`: corridas de selección baseline/smart.
- `outputs/final_patch_selection/`: corridas finales congeladas localmente.
- `outputs/segmentation/`: segmentación sobre patches seleccionados y
  comparación técnica de segmentación.
- `outputs/wsi_patches/`: extracciones WSI acotadas para debug.
- `outputs/model_checks/`: estados de carga de modelo.
- `outputs/inference_smoke/`: smoke tests de inferencia.
- `outputs/patches/`, `outputs/masks/`, `outputs/overlays/`,
  `outputs/metrics/`, `outputs/figures/`: estructura histórica/general.

Git debe versionar como máximo `.gitkeep` en estas carpetas, no resultados
pesados.

### `scripts/`

Entrypoints CLI del proyecto. La ruta principal actual está en scripts `01`,
`02`, `04`, `05`, `06`, `07`, `08` y `09`. El script `03` fue archivado.

### `src/`

Código fuente reusable. Contiene selección, inferencia, modelos,
preprocesamiento, visualización y soporte de evaluación.

### `src/selection/`

Módulos del flujo de selección de patches. Es core del proyecto.

### `src/inference/`

Módulos para carga/inferencia TIAToolbox, segmentación sobre patches
seleccionados y comparación de segmentación. Es core del proyecto.

### `src/models/`

Configuración y smoke test programático del modelo TIAToolbox BCSS. Aquí se
define el modelo `fcn_resnet50_unet-bcss`.

### `src/preprocessing/`

Utilidades de preprocesamiento, lectura WSI acotada y detección técnica de
tejido.

### `src/visualization/`

Funciones para overlays, leyendas y previews.

### `src/evaluation/`

Métricas futuras de evaluación con ground truth, como pixel accuracy e IoU. No
es el centro de los resultados actuales si no hay ground truth integrado.

## 3. Archivos raíz

### `README.md`

Documento operativo principal. Explica objetivo, alcance, estructura, ambientes,
comandos principales, etapas de selección/segmentación y advertencias sobre
datos y pesos.

### `LICENSE`

Licencia del repo.

### `.gitignore`

Evita versionar caches, entornos, datos, WSI, checkpoints y outputs generados.
Permite `.gitkeep` en carpetas de outputs para conservar estructura vacía sin
subir resultados.

### `environment.yml`

Ambiente Conda/Mamba principal. Es la vía recomendada para desarrollo local
porque coordina dependencias Python y librerías nativas como OpenSlide.

### `environment-linux-gpu.yml`

Ambiente alternativo para Linux con GPU. Debe validarse en el servidor real
según driver, CUDA disponible y políticas del entorno.

### `requirements.txt`

Respaldo pip. No es la fuente principal recomendada, pero ayuda a ver
dependencias Python base.

## 4. Scripts principales

### `scripts/01_check_environment.py`

Tipo: core de verificación.

Propósito: validar que el ambiente puede importar dependencias base y que las
carpetas esperadas existen.

Funciones relevantes:

- `check_imports()`
- `check_cuda()`
- `check_directories()`
- `main()`

Inputs: no requiere datos.  
Outputs: logs por consola; exit code `0` si el ambiente está usable.  
Uso en el flujo: primer comando de validación.

Ejemplo:

```bash
python scripts/01_check_environment.py
```

### `scripts/02_test_tiatoolbox_model.py`

Tipo: core de verificación de modelo.

Propósito: intentar cargar el modelo preentrenado de TIAToolbox y escribir un
JSON de estado.

Módulos internos:

- `src.models.tiatoolbox_bcss`

Inputs principales:

- `--model-name`, por defecto `fcn_resnet50_unet-bcss`
- `--device`
- `--output-json`

Outputs:

- `outputs/model_checks/tiatoolbox_bcss_model_status.json`

Este script valida carga del modelo. No corre inferencia final, no entrena y no
evalúa calidad.

### `scripts/04_run_inference.py`

Tipo: core de inferencia individual/smoke test.

Propósito: correr inferencia técnica sobre una imagen o patch individual usando
TIAToolbox.

Módulos internos:

- `src.inference.tiatoolbox_inference.run_inference_smoke_test`
- `src.models.tiatoolbox_bcss`

Inputs principales:

- `--image-path`
- `--model-name`
- `--device`
- `--input-mode patch|wsi`
- `--output-dir`
- `--overlay-alpha`

Outputs:

- `input_preview.png`
- `prediction_mask.png`
- `prediction_overlay.png`
- `prediction_overlay_with_legend.png`
- `legend.json`
- `legend.png`
- `inference_summary.json`

Uso en el flujo: smoke test antes de segmentar lotes de patches.

### `scripts/05_extract_wsi_patches.py`

Tipo: soporte WSI/debug.

Propósito: extraer un conjunto acotado y reproducible de patches level 0 desde
una WSI usando OpenSlide.

Módulos internos:

- `src.preprocessing.wsi_patch_extraction`

Inputs principales:

- `--wsi-path`
- `--output-dir`
- `--patch-size`
- `--max-patches`
- `--min-tissue-ratio`
- `--thumbnail-size`
- `--seed`

Outputs:

- `selected/`
- `patches_metadata.csv`
- `summary.json`
- `patch_selection_preview.png` si se solicita

Este script no es el selector formal actual, pero sirve como extractor/debug WSI
acotado y debe mantenerse.

### `scripts/05_evaluate_bcss.py`

Tipo: placeholder.

Propósito actual: imprimir una nota sobre evaluación futura con BCSS. Según la
estructura actual, todavía no ejecuta evaluación real ni descarga BCSS.

Se espera que una implementación futura conecte ground truth con métricas como
Dice, IoU/mIoU y pixel accuracy.

### `scripts/06_select_wsi_patches.py`

Tipo: core de selección de patches.

Propósito: entrada CLI para ejecutar `baseline_tiatoolbox`,
`smart_tissue_nuclei_v1`, `smart_tissue_nuclei_v2_light` o
`v3_server_quality` o `v4_embedding_assisted`.

Módulos internos:

- `src.selection.BaselineSelectionConfig`
- `src.selection.SmartTissueNucleiConfig`
- `src.selection.V3ServerQualityConfig`
- `src.selection.V4EmbeddingAssistedConfig`
- `src.selection.run_baseline_selection`
- `src.selection.run_smart_tissue_nuclei_selection`
- `src.selection.run_v3_server_quality_selection`
- `src.selection.run_v4_embedding_assisted_selection`

Inputs principales:

- `--wsi-path`
- `--output-dir`
- `--selector`
- `--patch-size`
- `--stride`
- `--max-patches`
- `--min-tissue-ratio`
- `--seed`
- parámetros smart como `--max-candidates-to-score`, `--feature-size`,
  `--nuclear-proxy`, `--spatial-strategy`, `--diversity-strategy`
- parámetros v3 como `--min-quality-score`, `--redundancy-penalty-weight`,
  `--cache-features`, `--resume` y `--output-mode`
- parámetros v4 como `--embedding-model-path`, `--embedding-device`,
  `--embedding-batch-size`, `--embedding-cluster-count` y
  `--reuse-embedding-cache`

Outputs:

- `selected/`
- `candidate_metadata.csv`
- `selected_metadata.csv`
- `selection_summary.json`
- `method_config.json`
- `patch_selection_preview.png`

Este script solo selecciona patches. No corre segmentación ni modelos deep
learning.

### `scripts/07_compare_patch_selectors.py`

Tipo: core de comparación de selectores.

Propósito: comparar dos carpetas de selección ya generadas.

Módulos internos:

- `src.selection.comparison.ComparisonConfig`
- `src.selection.comparison.compare_patch_selectors`

Inputs:

- `--baseline-dir`
- `--smart-dir`
- `--output-dir`
- `--feature-size`
- `--recompute-selected-features`

Outputs:

- `comparison_summary.json`
- `comparison_metrics.csv`
- `selected_overlap.csv`
- `comparison_selected_patches.csv`
- `comparison_preview.png`
- `comparison_preview_selected_only.png`
- `comparison_notes.md`

No corre segmentación y no evalúa ground truth.

### `scripts/08_segment_selected_patches.py`

Tipo: core de segmentación sobre patches seleccionados.

Propósito: tomar una carpeta generada por `scripts/06_select_wsi_patches.py`,
leer `selected_metadata.csv` y correr inferencia sobre los PNG en `selected/`.

Módulos internos:

- `src.inference.selected_patch_segmentation`
- `src.models.tiatoolbox_bcss`

Inputs:

- `--input-selection-dir`
- `--output-dir`
- `--model-name`
- `--device`
- `--input-mode`
- `--overlay-alpha`
- `--limit-patches`

Outputs:

- `per_patch/`
- `masks/`
- `overlays/`
- `overlays_with_legend/`
- `input_previews/`
- `per_patch_segmentation.csv`
- `inference_summary.json`
- `method_config.json`

No repite selección y no compara métodos.

### `scripts/09_compare_segmentation_on_selected_patches.py`

Tipo: core de comparación técnica de segmentación.

Propósito: comparar dos carpetas de segmentación ya generadas, por ejemplo
baseline vs smart v2.

Módulos internos:

- `src.inference.segmentation_comparison`

Inputs:

- `--baseline-seg-dir`
- `--smart-seg-dir`
- `--output-dir`
- `--max-preview-patches`
- `--preview-source`

Outputs:

- `segmentation_comparison_summary.json`
- `segmentation_comparison_metrics.csv`
- `segmentation_class_distribution.csv`
- `segmentation_patch_rows.csv`
- `segmentation_comparison_preview.png`
- `segmentation_comparison_notes.md`

No re-ejecuta segmentación y no usa ground truth.

## 5. Código fuente en `src/`

### `src/selection/`

#### `candidate_generation.py`

Responsabilidad: generar candidatos desde WSI a partir de una grilla level 0 y
un thumbnail con máscara técnica de tejido.

Elementos importantes:

- `PatchCandidate`
- `generate_grid_coordinates()`
- `thumbnail_bbox_for_patch()`
- `compute_thumbnail_tissue_ratio()`
- `generate_tissue_candidates()`

Participa en `scripts/06_select_wsi_patches.py` a través de los selectores
baseline y smart.

#### `tiatoolbox_baseline.py`

Responsabilidad: implementar el selector `baseline_tiatoolbox`.

Elementos importantes:

- `BaselineSelectionConfig`
- `run_baseline_selection()`
- constantes como `BASELINE_SELECTOR_NAME`, `CANDIDATE_POOL`,
  `CANDIDATE_METADATA_SEMANTICS`

Este módulo genera el pool de candidatos, evalúa patches reales hasta cumplir
`max_patches`, guarda metadata y preview. Es un baseline técnico tipo
TIAToolbox, no una llamada completa a una API oficial de TIAToolbox para todo
el flujo.

#### `smart_tissue_nuclei.py`

Responsabilidad: implementar `smart_tissue_nuclei_v1` y
`smart_tissue_nuclei_v2_light`.

Elementos importantes:

- `SmartTissueNucleiConfig`
- `run_smart_tissue_nuclei_selection()`
- `SMART_SELECTOR_NAME`
- `SMART_V2_LIGHT_SELECTOR_NAME`

Lee el mismo pool thumbnail-filtered, calcula features en una muestra de
candidatos y selecciona bajo un presupuesto fijo.

#### `scoring.py`

Responsabilidad: normalizar features y calcular `score_raw`.

Elementos importantes:

- `DEFAULT_SMART_WEIGHTS`
- `normalize_feature()`
- `apply_feature_scores()`

Los pesos actuales combinan `tissue_ratio`, `nuclear_signal`, `visual_entropy`,
`blur_score` y `artifact_penalty`.

#### `quality_filters.py`

Responsabilidad: calcular features visuales livianas para smart selectors.

Elementos importantes:

- `compute_rgb_purple_nuclear_signal()`
- `compute_hed_nuclear_signal()`
- `compute_nuclear_signal()`
- `compute_visual_entropy()`
- `compute_blur_score()`
- `compute_artifact_penalty()`
- `compute_patch_features()`

Estas features son proxies técnicos. No detectan tumor ni reemplazan revisión
histopatológica.

#### `diversity.py`

Responsabilidad: manejar diversidad espacial y por features.

Elementos importantes:

- `patch_center()`
- `proximity_penalty()`
- `parse_quota_grid()`
- `assign_spatial_regions()`
- `feature_diversity_bonus()`
- `greedy_select_with_spatial_penalty()`
- `select_with_spatial_quotas()`

`v1` usa penalización espacial greedy. `v2_light` puede usar cuotas espaciales
suaves y diversidad por features.

#### `manifests.py`

Responsabilidad: definir columnas estables y escribir CSV/JSON.

Elementos importantes:

- `CANDIDATE_METADATA_FIELDS`
- `SELECTED_METADATA_FIELDS`
- `write_csv_manifest()`
- `write_json_manifest()`

Este archivo mantiene consistencia de outputs entre baseline y smart.

#### `previews.py`

Responsabilidad: dibujar previews de selección sobre thumbnails.

Elementos importantes:

- `_row_to_patch_box()`
- `save_wsi_patch_selection_preview()`

Usa `src.visualization.patch_preview.PatchBox`.

#### `comparison.py`

Responsabilidad: comparar corridas de selección.

Elementos importantes:

- `SelectorRun`
- `ComparisonConfig`
- `load_selector_run()`
- `validate_shared_config()`
- `compute_overlap_metrics()`
- `recompute_selected_patch_features()`
- `compute_spatial_metrics()`
- `build_comparison_metrics_rows()`
- `save_comparison_preview()`
- `save_selected_only_comparison_preview()`
- `write_comparison_notes()`
- `compare_patch_selectors()`

Este módulo produce métricas técnicas de comparación entre carpetas de
selección.

#### `__init__.py`

Responsabilidad: exportar la API pública de selección usada por `scripts/06`.
Expone configs, nombres de selectores y runners.

### `src/inference/`

#### `tiatoolbox_inference.py`

Responsabilidad: correr inferencia controlada con TIAToolbox y transformar
salidas en máscaras, overlays, leyendas y summaries.

Elementos importantes:

- `run_inference_smoke_test()`
- `clear_output_dir_safely()`
- `discover_class_mapping()`
- `build_class_legend()`
- `write_inference_summary()`
- `write_legend_json()`

Maneja `input_mode=patch` y `input_mode=wsi`, obtiene predicciones desde el
output de `SemanticSegmentor`, genera leyendas y registra warnings. Es usado por
`scripts/04` y por la segmentación batch de `scripts/08`.

#### `selected_patch_segmentation.py`

Responsabilidad: aplicar segmentación a todos los patches seleccionados por una
corrida de selección.

Elementos importantes:

- `SelectedPatchSegmentationConfig`
- `segment_selected_patches()`

Valida que existan `selected_metadata.csv`, `selection_summary.json`,
`method_config.json` y `selected/`. Procesa patches válidos, continúa ante
errores por patch, copia outputs globales y escribe
`per_patch_segmentation.csv` e `inference_summary.json`.

#### `segmentation_comparison.py`

Responsabilidad: comparar dos corridas de segmentación ya generadas.

Elementos importantes:

- `SegmentationComparisonConfig`
- `RunData`
- `compare_segmentation_runs()`

Calcula métricas operativas, distribución de clases predichas, filas por patch,
preview comparativo y notas Markdown. No re-ejecuta inferencia.

#### `run_tiatoolbox_baseline.py`

Tipo: placeholder histórico.

Contiene `TARGET_MODEL_NAME = "fcn_resnet50_unet-bcss"` y
`describe_baseline()`. Según la estructura actual, no participa en el pipeline
principal.

#### `__init__.py`

Inicializador del paquete `src.inference`.

### `src/models/`

#### `tiatoolbox_bcss.py`

Responsabilidad: configurar, validar y cargar el modelo preentrenado de
TIAToolbox.

Elementos importantes:

- `DEFAULT_MODEL_NAME = "fcn_resnet50_unet-bcss"`
- `SUPPORTED_DEVICES`
- `resolve_torch_device()`
- `build_model_status()`
- `write_model_status_json()`

Este modelo se usa para segmentación semántica posterior. No se usa para
seleccionar patches en baseline ni smart selectors.

#### `__init__.py`

Inicializador del paquete `src.models`.

### `src/preprocessing/`

#### `wsi_patch_extraction.py`

Responsabilidad: soporte WSI con OpenSlide y funciones compartidas de tejido.

Elementos importantes:

- `WsiPatchExtractionConfig`
- `WsiPatchCandidate`
- `estimate_thumbnail_tissue_mask()`
- `compute_simple_tissue_ratio()`
- `clear_output_dir_safely()`
- `extract_wsi_patches()`

Lo usan `scripts/05_extract_wsi_patches.py` y módulos de selección. La máscara
técnica de tejido usa una regla simple: media RGB menor a 235 y desviación
estándar mayor a 8.

#### `tissue_detection.py`

Responsabilidad: máscara simple de tejido para imágenes RGB. Define
`estimate_tissue_mask()`, basada en umbral contra fondo claro. Es un baseline
técnico, no un detector clínico.

#### `__init__.py`

Inicializador del paquete.

### `src/visualization/`

#### `segmentation_overlay.py`

Responsabilidad: transformar máscaras de clases en visualizaciones.

Elementos importantes:

- `color_for_class_id()`
- `normalize_label_mask()`
- `resize_label_mask()`
- `colorize_label_mask()`
- `overlay_label_mask()`
- `render_class_legend_image()`
- `append_legend_to_image()`

Usa paleta para la salida agrupada del modelo BCSS de TIAToolbox.

#### `patch_preview.py`

Responsabilidad: dibujar rectángulos de selección sobre imágenes/thumbnail.

Elementos importantes:

- `PatchBox`
- `save_patch_selection_preview()`

Aunque nació para previews simples, sigue siendo soporte activo porque lo usan
`src/selection/previews.py` y `src/preprocessing/wsi_patch_extraction.py`.

#### `overlays.py`

Según la estructura actual, helper simple no usado por el pipeline core. Define
`create_overlay()` para pintar en rojo píxeles no cero de una máscara binaria o
entera.

#### `__init__.py`

Inicializador del paquete.

### `src/evaluation/`

#### `metrics.py`

Responsabilidad: soporte futuro para evaluación con ground truth.

Elementos importantes:

- `pixel_accuracy()`
- `mean_iou()`

Según la estructura actual, no está integrado en los resultados principales
porque aún no hay flujo formal con ground truth BCSS.

#### `__init__.py`

Inicializador del paquete.

### `src/io/`

#### `slide_reader.py`

Según la estructura actual, helper liviano no usado por el pipeline core.
Define `read_image()` para cargar imágenes comunes con PIL.

## 6. Selectores de patches

### 6.1 `baseline_tiatoolbox`

`baseline_tiatoolbox` formaliza un baseline técnico tipo TIAToolbox para
selección de patches desde WSI. Su objetivo es ser reproducible, simple y
comparable contra los selectores propios.

Qué hace:

- genera una grilla determinística sobre la WSI;
- crea un thumbnail;
- estima una máscara técnica de tejido;
- filtra candidatos por `thumbnail_tissue_ratio >= min_tissue_ratio`;
- baraja candidatos con `seed`;
- lee patches reales uno por uno hasta seleccionar `max_patches`;
- calcula `tissue_ratio` real solo para candidatos evaluados;
- guarda los patches seleccionados en `selected/`.

Qué no hace:

- no usa deep learning;
- no usa `fcn_resnet50_unet-bcss`;
- no segmenta;
- no detecta tumor;
- no decide relevancia clínica.

Metadata producida:

- `candidate_metadata.csv`: pool común de candidatos filtrados por thumbnail.
  Incluye candidatos evaluados y no evaluados.
- `selected_metadata.csv`: solo patches seleccionados.
- `selection_summary.json`: contadores y resumen de corrida.
- `method_config.json`: configuración experimental.
- `patch_selection_preview.png`: preview técnico sobre thumbnail.

Parámetros relevantes:

- `patch_size`: tamaño del patch level 0.
- `stride`: paso de la grilla.
- `max_patches`: presupuesto de selección.
- `min_tissue_ratio`: umbral técnico de tejido.
- `thumbnail_max_size`: tamaño máximo del thumbnail.
- `seed`: reproducibilidad del shuffle.
- `overwrite`: regeneración segura del output dir.

### 6.2 `smart_tissue_nuclei_v1`

`smart_tissue_nuclei_v1` es el primer selector propio liviano. Usa el mismo pool
de candidatos thumbnail-filtered que el baseline, pero asigna score a una
muestra de candidatos antes de seleccionar.

Qué hace:

- usa `max_candidates_to_score` para no leer todo el pool si es grande;
- usa `feature_size` para calcular features en una versión reducida del patch;
- calcula proxies visuales;
- normaliza features;
- calcula `score_raw`;
- aplica selección greedy con penalización espacial.

Features principales:

- `tissue_ratio`
- `nuclear_signal` con proxy RGB púrpura/hematoxilina aproximada
- `visual_entropy`
- `blur_score`
- `artifact_penalty`
- `spatial_penalty`
- `score_raw`
- `score_final`

Qué no hace:

- no usa deep learning;
- no usa el modelo de segmentación;
- no clasifica tumor;
- no valida clínicamente los patches.

Rol actual: versión intermedia/ablation para comparar contra baseline y
`smart_tissue_nuclei_v2_light`.

### 6.3 `smart_tissue_nuclei_v2_light`

`smart_tissue_nuclei_v2_light` es el selector propio candidato final actual.
Mantiene la filosofía liviana y reproducible de v1, pero mejora tres aspectos:

- proxy nuclear por HED color deconvolution;
- cuotas espaciales suaves;
- diversidad por features.

HED en este contexto:

- es una aproximación computacional para separar señales de tinción H&E;
- se usa para obtener una señal técnica de hematoxilina/núcleos;
- no implica diagnóstico ni detección clínica.

Parámetros típicos:

- `nuclear_proxy = hed_deconvolution`
- `spatial_strategy = quotas`
- `quota_grid = 4x4`
- `quota_min_score_quantile`
- `diversity_strategy = farthest_feature`
- `feature_diversity_weight`

Por qué se considera candidato final actual:

- estima mejor la señal nuclear que el proxy RGB simple;
- mantiene una cobertura espacial más controlada;
- reduce concentración en zonas cercanas;
- sigue siendo CPU-friendly;
- no introduce dependencias de modelos deep learning para selección.

### 6.4 `v3_server_quality`

`v3_server_quality` es un selector orientado a ejecución en servidor/iHealth.
Se implementa separado de `smart_tissue_nuclei.py` en
`src/selection/v3_server_quality.py` para evitar que el selector CPU-friendly
crezca demasiado.

Qué hace:

- usa el mismo pool thumbnail-filtered que los demás selectores;
- puede scorear más candidatos por corrida;
- usa `feature_size` mayor por defecto;
- calcula proxies técnicos de calidad segmentable, celularidad HED/RGB,
  heterogeneidad y utilidad esperada;
- aplica cuotas espaciales suaves;
- aplica diversidad por features;
- agrega penalización de redundancia;
- puede escribir `scored_candidates.csv` como cache/debug.

Qué no hace:

- no usa deep learning;
- no usa `fcn_resnet50_unet-bcss` para seleccionar;
- no detecta tumor con certeza clínica;
- no calcula RCB;
- no valida clínicamente los patches.

Rol actual: selector server-quality no model-assisted. Sirve como base técnica
para `v4_embedding_assisted`.

### 6.5 `v4_embedding_assisted`

`v4_embedding_assisted` extiende `v3_server_quality` con embeddings UNI. El
módulo principal es `src/selection/v4_embedding_assisted.py` y el soporte de
embeddings vive en `src/selection/embedding_scoring.py`.

Qué agrega:

- carga modular de UNI desde pesos/modelo local;
- cache de embeddings en `embedding_cache.npz`;
- metadata de cache en `embedding_cache_metadata.json`;
- clustering morfológico de candidatos scoreados;
- balance de clusters;
- representatividad respecto a centroides;
- penalización por redundancia morfológica;
- bonus de diversidad por embedding.

Qué no hace:

- no usa `fcn_resnet50_unet-bcss` para seleccionar;
- no ejecuta segmentación preliminar;
- no descarga pesos automáticamente;
- no guarda tokens;
- no usa UNI como detector clínico;
- no calcula RCB ni valida clínicamente.

Si UNI no está disponible y no existe cache compatible, el selector debe fallar
con un mensaje accionable en vez de simular embeddings.

En el entorno local actual se validó la integración y la falla limpia sin pesos
UNI. La ejecución real con UNI requiere proporcionar `--embedding-model-path` o
un cache compatible generado con el mismo backend, modelo, métrica, dimensión y
conjunto de candidatos.

## 7. Modelo de segmentación TIAToolbox

El modelo usado es:

```text
fcn_resnet50_unet-bcss
```

Este modelo viene de TIAToolbox como modelo preentrenado asociado a BCSS. En
este repo se usa para segmentación semántica técnica de patches, no para
seleccionar patches.

Aparece en:

- `scripts/02_test_tiatoolbox_model.py`
- `scripts/04_run_inference.py`
- `scripts/08_segment_selected_patches.py`
- `src/models/tiatoolbox_bcss.py`
- `src/inference/tiatoolbox_inference.py`

Qué produce:

- máscara de clases predichas por el modelo;
- overlay;
- overlay con leyenda;
- preview de entrada;
- `legend.json`;
- `inference_summary.json`;
- `class_pixel_counts`;
- `unique_prediction_values`;
- warnings técnicos.

Clases y advertencia de interpretación:

- La salida agrupada del modelo TIAToolbox usa cinco clases del ejemplo BCSS
  agrupado: `0 = Tumour`, `1 = Stroma`, `2 = Inflammatory`, `3 = Necrosis`,
  `4 = Others`.
- Eso no debe mezclarse con los códigos raw de BCSS, donde `0` significa
  `outside_roi / don't care`.
- Las clases predichas son outputs técnicos del modelo, no ground truth clínico.

Diferencia de resolución:

- El patch visual puede ser `1024x1024`.
- La máscara cruda puede ser `512x512`.
- Para el overlay, la máscara se reescala con nearest neighbor para preservar
  etiquetas discretas.
- `class_pixel_counts` corresponde a la resolución cruda de predicción, no
  necesariamente al overlay visual.

## 8. Flujos del proyecto

### Flujo A: verificar ambiente y modelo

Comandos:

```bash
python scripts/01_check_environment.py
python scripts/02_test_tiatoolbox_model.py
```

`01_check_environment.py` valida imports, CUDA y carpetas.  
`02_test_tiatoolbox_model.py` valida carga del modelo TIAToolbox y escribe un
JSON de estado.

### Flujo B: selección de patches desde WSI

Comando base:

```bash
python scripts/06_select_wsi_patches.py ...
```

Entrada:

- WSI (`.svs`, `.tif`, `.tiff`, `.ndpi`, etc.).

Salida:

- carpeta de selección con `candidate_metadata.csv`,
  `selected_metadata.csv`, `selection_summary.json`, `method_config.json`,
  `selected/` y preview.

Diferencia clave:

- `candidate_metadata.csv`: pool común thumbnail-filtered.
- `selected_metadata.csv`: solo patches finalmente seleccionados.

### Flujo C: comparación de selectores

Comando:

```bash
python scripts/07_compare_patch_selectors.py ...
```

Compara dos carpetas de selección. No corre segmentación. Calcula overlap,
features recomputadas, diversidad espacial, métricas y previews.

### Flujo D: segmentación sobre patches seleccionados

Comando:

```bash
python scripts/08_segment_selected_patches.py ...
```

Lee `selected_metadata.csv` y `selected/`, corre el modelo de segmentación sobre
los patches, genera máscaras, overlays, CSV por patch y summary global. No
repite selección.

### Flujo E: comparación técnica de segmentación

Comando:

```bash
python scripts/09_compare_segmentation_on_selected_patches.py ...
```

Compara outputs de segmentación baseline vs smart. No re-ejecuta segmentación.
Compara distribución de clases predichas, warnings, runtime y métricas
técnicas. No usa ground truth.

### Flujo F: legacy small-image pipeline

Ubicación:

```text
archive/legacy_small_image_pipeline/
```

Es un flujo archivado para imágenes pequeñas. No es ruta principal. Sirve como
referencia histórica.

## 9. Qué modelos se usan realmente

| componente | usa modelo deep learning | modelo usado | para qué se usa | qué NO hace |
|---|---|---|---|---|
| `baseline_tiatoolbox` | No | Ninguno | selección técnica por grilla, thumbnail y tejido | no segmenta, no usa DL, no diagnostica |
| `smart_tissue_nuclei_v1` | No | Ninguno | selección técnica con features/proxies visuales | no usa segmentación, no detecta tumor |
| `smart_tissue_nuclei_v2_light` | No | Ninguno | selección técnica con HED proxy, cuotas y diversidad | no usa DL, no valida clínicamente |
| `v3_server_quality` | No | Ninguno | selección técnica server-quality con proxies de utilidad, diversidad y baja redundancia | no usa modelo de segmentación, no diagnostica |
| `v4_embedding_assisted` | Sí, solo como extractor de embeddings | UNI | representación morfológica para diversidad, clusters y reducción de redundancia | no usa segmentación para seleccionar, no diagnostica |
| TIAToolbox segmentation | Sí | `fcn_resnet50_unet-bcss` | segmentación semántica posterior sobre patches seleccionados | no selecciona patches, no produce ground truth clínico |
| Evaluación BCSS futura | No implementado como flujo completo | Pendiente | comparar predicción vs ground truth cuando exista integración | no forma parte central de resultados actuales |

## 10. Outputs y artefactos

Los outputs no se versionan. `.gitkeep` existe para mantener la estructura de
carpetas vacías, pero no habilita subir resultados reales.

Carpetas importantes:

- `outputs/patch_selection/`: corridas de selección y comparación de
  selectores.
- `outputs/final_patch_selection/`: resultados finales locales de selección.
- `outputs/segmentation/`: segmentación batch y comparación técnica.
- `outputs/wsi_patches/`: extracciones WSI acotadas.
- `outputs/model_checks/`: estado de carga del modelo.
- `outputs/inference_smoke/`: smoke tests.

Artefactos importantes:

- `candidate_metadata.csv`
- `selected_metadata.csv`
- `selection_summary.json`
- `method_config.json`
- `per_patch_segmentation.csv`
- `inference_summary.json`
- `segmentation_comparison_summary.json`
- overlays y previews locales

No subir a Git:

- WSI;
- patches generados;
- máscaras;
- overlays;
- outputs de comparación;
- checkpoints/pesos;
- caches de modelos;
- datos clínicos sensibles.

## 11. Limitaciones actuales

- No hay ground truth integrado en el flujo principal actual.
- La comparación de segmentación es técnica, no clínica.
- La segmentación depende de un modelo preentrenado.
- Los selectores smart usan proxies visuales, no verificación patológica.
- Métricas como Dice/IoU quedan para cuando exista ground truth compatible.
- La máscara cruda del modelo puede tener resolución distinta al overlay.
- HED y señal nuclear son proxies técnicos, no diagnóstico.
- Según la estructura actual, `src/io/slide_reader.py` y
  `src/visualization/overlays.py` existen pero no están conectados al pipeline
  core.
- `scripts/05_evaluate_bcss.py` y
  `src/inference/run_tiatoolbox_baseline.py` son placeholders/históricos.

## 12. Glosario breve

- WSI: Whole Slide Image, imagen histológica completa escaneada.
- patch: recorte rectangular de una WSI.
- thumbnail: versión reducida de una WSI usada para estimar tejido sin leer
  toda la imagen a resolución completa.
- tissue mask: máscara técnica que estima dónde hay tejido.
- candidate pool: conjunto de candidatos que pasan el filtro inicial de tejido.
- selected patches: subconjunto final de patches elegido por un selector.
- baseline: método simple usado como referencia comparativa.
- smart selector: selector propio con scoring por features/proxies.
- HED: aproximación de color deconvolution para separar señales H&E.
- segmentation mask: matriz de clases predichas por el modelo.
- overlay: visualización de la máscara sobre el patch RGB.
- class_pixel_counts: conteo de píxeles por clase predicha en la máscara cruda.
- ground truth: anotación de referencia usada para evaluar métricas.
- BCSS: Breast Cancer Semantic Segmentation dataset.
- TIAToolbox: toolkit de patología digital usado para modelo e inferencia.
- RCB: Residual Cancer Burden; este proyecto no lo calcula.

## 13. Mapa rápido para saber dónde tocar código

| Quiero cambiar... | Archivo(s) |
|---|---|
| cómo se generan candidatos | `src/selection/candidate_generation.py`, `src/preprocessing/wsi_patch_extraction.py` |
| baseline | `src/selection/tiatoolbox_baseline.py`, `scripts/06_select_wsi_patches.py` |
| smart selector | `src/selection/smart_tissue_nuclei.py`, `scripts/06_select_wsi_patches.py` |
| scoring/features | `src/selection/scoring.py`, `src/selection/quality_filters.py`, `src/selection/diversity.py` |
| comparación de selectores | `src/selection/comparison.py`, `scripts/07_compare_patch_selectors.py` |
| inferencia TIAToolbox | `src/inference/tiatoolbox_inference.py`, `scripts/04_run_inference.py` |
| segmentación sobre selected patches | `src/inference/selected_patch_segmentation.py`, `scripts/08_segment_selected_patches.py` |
| comparación de segmentación | `src/inference/segmentation_comparison.py`, `scripts/09_compare_segmentation_on_selected_patches.py` |
| overlays/leyendas | `src/visualization/segmentation_overlay.py` |
| previews de selección | `src/selection/previews.py`, `src/visualization/patch_preview.py` |
| evaluación con ground truth | `src/evaluation/metrics.py`, futuro reemplazo/implementación de `scripts/05_evaluate_bcss.py` |
| soporte WSI acotado/debug | `scripts/05_extract_wsi_patches.py`, `src/preprocessing/wsi_patch_extraction.py` |

## 14. Lectura recomendada para nuevos integrantes

Orden sugerido:

1. `README.md`
2. `docs/maintenance/repo_cleanup_audit.md`
3. `scripts/06_select_wsi_patches.py`
4. `src/selection/tiatoolbox_baseline.py`
5. `src/selection/smart_tissue_nuclei.py`
6. `src/selection/comparison.py`
7. `scripts/08_segment_selected_patches.py`
8. `src/inference/tiatoolbox_inference.py`
9. `src/inference/segmentation_comparison.py`

Con eso se entiende la ruta principal sin entrar primero en el flujo histórico
de imágenes pequeñas.
