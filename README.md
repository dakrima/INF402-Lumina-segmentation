# INF402 - Lumina Segmentation

Repositorio del proyecto de INF402 - Procesamiento y Reconstrucción de Imágenes Médicas, articulado con Lumina. El objetivo es construir una base reproducible para selección inteligente de patches y segmentación semántica en imágenes histopatológicas H&E de cáncer de mama.

## Objetivo

El proyecto busca apoyar el análisis visual de imágenes histopatológicas mediante un pipeline computacional que reduzca regiones vacías o poco informativas, seleccione patches útiles y genere máscaras, overlays o mapas de probabilidad revisables.

La salida del grupo es un insumo técnico para análisis posterior. El sistema debe entenderse como apoyo académico/computacional y no como herramienta clínica autónoma.

## Alcance técnico

El foco del repositorio es:

- tissue detection;
- patch filtering;
- selección inteligente de patches;
- segmentación semántica posterior;
- generación de máscaras y overlays;
- evaluación con métricas de segmentación cuando exista ground truth;
- posible fine-tuning si el baseline preentrenado no basta.

Estado actual para selección de patches: el flujo principal del paper compara `baseline_tiatoolbox` contra `v4_1_medical_embedding_assisted`. `smart_tissue_nuclei_v1`, `smart_tissue_nuclei_v2_light`, `v3_server_quality` y `v4_embedding_assisted` se conservan como iteraciones previas, ablations o soporte interno para reproducibilidad.

## Lo que este proyecto no hace

Este proyecto no:

- diagnostica cáncer;
- reemplaza al patólogo;
- calcula RCB completo;
- cuantifica cáncer residual como objetivo principal del grupo;
- promete validación clínica definitiva;
- promete descartar tejido sano con certeza clínica;
- sube datasets, WSI, checkpoints ni outputs pesados al repositorio.

## Ruta técnica

```text
WSI / imagen histopatológica H&E
  -> tissue detection
  -> baseline TIAToolbox: ventana deslizante + máscara Otsu + min_mask_ratio
  -> selector propio de patches
  -> segmentación semántica posterior
  -> máscara/overlay revisable
  -> evaluación técnica y posible adaptación
```

La estrategia inicial de INF402 es formalizar un baseline real basado en TIAToolbox y compararlo luego contra un selector propio de patches. La segmentación con `fcn_resnet50_unet-bcss` se mantiene como etapa posterior para generar máscaras/overlays revisables. Fine-tuning queda como opción posterior si la segmentación no es suficiente; entrenar desde cero no es la primera opción.

## Flujo principal de selección de patches

Para el flujo principal del paper se consideran dos métodos:

1. `baseline_tiatoolbox`
   - Baseline real basado en TIAToolbox.
   - Usa `SlidingWindowPatchExtractor`, `input_mask="otsu"` y `min_mask_ratio`.
   - No usa segmentación, UNI, embeddings ni features médicas para seleccionar.

2. `v4_1_medical_embedding_assisted`
   - Selector técnico propuesto.
   - Usa proxies técnicos de imagen médica y embeddings UNI como reranking morfológico.
   - No usa segmentación para seleccionar, no calcula RCB y no emite diagnóstico.

`smart_tissue_nuclei_v1`, `smart_tissue_nuclei_v2_light`, `v3_server_quality` y `v4_embedding_assisted` siguen disponibles como métodos legacy/experimentales para trazabilidad y reproducibilidad. `v4_1_medical_embedding_assisted` puede reutilizar funciones internas de `v3_server_quality` para scoring/base técnica; esto no implica que `v3_server_quality` se ejecute como método principal.

Los métodos no comparten necesariamente el mismo pool inicial de candidatos. El baseline utiliza el pool generado por TIAToolbox con máscara Otsu, mientras que v4.1 utiliza su propia generación de candidatos y posterior ranking técnico. Esta diferencia debe reportarse como parte de la configuración experimental.

## Estructura del repositorio

```text
.
├── README.md
├── environment.yml
├── environment-linux-gpu.yml
├── requirements.txt
├── archive/
│   └── legacy_small_image_pipeline/
├── docs/
│   ├── parte_1/
│   ├── parte_2/
│   └── maintenance/
├── data/
│   ├── raw/
│   ├── external/
│   ├── interim/
│   └── processed/
├── outputs/
│   ├── patches/
│   ├── masks/
│   ├── overlays/
│   ├── metrics/
│   ├── figures/
│   ├── patch_selection/
│   ├── final_patch_selection/
│   ├── segmentation/
│   ├── wsi_patches/
│   ├── model_checks/
│   └── inference_smoke/
├── notebooks/
├── scripts/
└── src/
```

`data/` y `outputs/` se mantienen con `.gitkeep` cuando corresponde, pero su contenido real está ignorado por Git. Los outputs locales, WSI, checkpoints, patches, máscaras, overlays y métricas pesadas no deben subirse al repositorio.

## Ambientes reproducibles

Se usa Conda/Mamba como primera opción porque permite coordinar dependencias Python y librerías nativas como OpenSlide de forma más controlada entre macOS, Linux y servidores con GPU.

`requirements.txt` queda como respaldo pip, no como fuente principal recomendada.

## Por qué no partir con Docker

Docker puede ser útil para reproducibilidad final, pero puede agregar fricción inicial en servidores con GPU, permisos, drivers NVIDIA, montaje de datos grandes y entornos HPC donde a veces se usa Apptainer/Singularity. Primero se validará el pipeline con Conda/Mamba; luego se evaluará Docker o Apptainer si el flujo ya funciona.

## Instalación local

```bash
mamba env create -f environment.yml
mamba activate inf402-lumina-seg
python scripts/01_check_environment.py
```

Si no tienes `mamba`, puedes usar `conda env create -f environment.yml`.

## Instalación en Linux GPU

Antes de crear el ambiente, revisar:

```bash
nvidia-smi
```

Luego:

```bash
mamba env create -f environment-linux-gpu.yml
mamba activate inf402-lumina-seg
nvidia-smi
python scripts/01_check_environment.py
```

La combinación PyTorch/CUDA en `environment-linux-gpu.yml` debe validarse en el servidor real. En iHealth o NLHPC puede requerir ajuste según driver, CUDA disponible y política local.

## Verificación de ambiente

```bash
python scripts/01_check_environment.py
```

El script revisa importaciones base, disponibilidad de CUDA en PyTorch y existencia de carpetas esperadas.

## Flujo legacy / histórico para imágenes pequeñas

El flujo inicial para extraer patches desde imágenes pequeñas fue archivado en:

```text
archive/legacy_small_image_pipeline/
```

Ese flujo conserva `scripts/03_extract_patches.py`, `src/patching/`, `src/config/` y el notebook de exploración como referencia histórica. No forma parte del pipeline principal actual basado en WSI, selección inteligente de patches y segmentación técnica posterior. Si se quiere ejecutar desde `archive/`, puede requerir ajustes de imports y paths.

## Extracción reproducible desde WSI con OpenSlide

Para extraer un conjunto acotado de patches `1024x1024` desde una WSI `.svs`, usar:

```bash
python scripts/05_extract_wsi_patches.py \
  --wsi-path /Users/davidkripper/demoCasesMvpFeria/TCGA-A2-A3XS-01Z-00-DX1.867925C0-91D8-40A0-9FEA-25A635AC31E7.svs \
  --output-dir outputs/wsi_patches/test_tcga_a2_a3xs \
  --patch-size 1024 \
  --max-patches 8 \
  --min-tissue-ratio 0.2 \
  --thumbnail-size 2048 \
  --clear-output \
  --preview-image
```

El script usa OpenSlide, lee metadata de la WSI, crea un thumbnail, estima una máscara simple de tejido con `mean < 235` y `std > 8`, selecciona coordenadas level 0 de forma reproducible y guarda patches aceptados en `selected/`. También escribe `patches_metadata.csv`, `summary.json` y, si se solicita, `patch_selection_preview.png`.

Este paso solo hace selección y extracción técnica de patches. No corre inferencia, no evalúa calidad, no diagnostica, no calcula RCB y no constituye validación clínica. Los outputs quedan bajo `outputs/` y no deben subirse a Git.

Para correr inferencia sobre un patch generado:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python scripts/04_run_inference.py \
  --image-path outputs/wsi_patches/test_tcga_a2_a3xs/selected/patch_0000_x12345_y67890.png \
  --model-name fcn_resnet50_unet-bcss \
  --device cpu \
  --input-mode patch \
  --output-dir outputs/inference_smoke/test_wsi_patch_0000 \
  --clear-output
```

## Etapa 1 - baseline_tiatoolbox

El selector formal de Etapa 1 vive en `scripts/06_select_wsi_patches.py` y genera una corrida reproducible de selección baseline sobre WSI:

```bash
/Users/davidkripper/miniforge3/envs/inf402-lumina-seg/bin/python scripts/06_select_wsi_patches.py \
  --wsi-path /Users/davidkripper/demoCasesMvpFeria/TCGA-A2-A3XS-01Z-00-DX1.867925C0-91D8-40A0-9FEA-25A635AC31E7.svs \
  --output-dir outputs/patch_selection/baseline_tiatoolbox_tcga_a2_a3xs \
  --selector baseline_tiatoolbox \
  --patch-size 1024 \
  --stride 1024 \
  --max-patches 16 \
  --min-tissue-ratio 0.20 \
  --seed 42 \
  --overwrite
```

La salida incluye `selected/`, `candidate_metadata.csv`, `selected_metadata.csv`, `selection_summary.json`, `method_config.json` y `patch_selection_preview.png` cuando la preview se puede generar. `candidate_metadata.csv` contiene el pool de candidatos generado por `SlidingWindowPatchExtractor` con máscara Otsu y `min_mask_ratio`, mientras que `selected_metadata.csv` contiene solo los patches seleccionados.

Este baseline no usa ranking inteligente, señal nuclear, HED, diversidad espacial, HoVer-Net, CLAM, active learning ni embeddings. La separación entre pool de candidatos y seleccionados mantiene trazabilidad experimental bajo el mismo presupuesto de patches, sin afirmar diagnóstico, RCB ni validación clínica.

## Legacy / experimental - smart_tissue_nuclei_v1

`smart_tissue_nuclei_v1` es una iteración previa/ablation. Usa un pool de candidatos filtrados por thumbnail y scorea una muestra reproducible de candidatos con features simples: `tissue_ratio`, `nuclear_signal`, `visual_entropy`, `blur_score`, `artifact_penalty` y penalización espacial greedy.

```bash
conda run -n inf402-lumina-seg python scripts/06_select_wsi_patches.py \
  --wsi-path /Users/davidkripper/demoCasesMvpFeria/TCGA-A2-A3XS-01Z-00-DX1.867925C0-91D8-40A0-9FEA-25A635AC31E7.svs \
  --output-dir outputs/patch_selection/smart_tcga_a2_a3xs \
  --selector smart_tissue_nuclei_v1 \
  --patch-size 1024 \
  --stride 1024 \
  --max-patches 16 \
  --min-tissue-ratio 0.20 \
  --seed 42 \
  --max-candidates-to-score 300 \
  --feature-size 256 \
  --lambda-spatial 0.15 \
  --overwrite
```

`--max-candidates-to-score` y `--feature-size` mantienen el flujo CPU-friendly: los patches se leen uno por uno, las features se calculan sobre una versión reducida y solo se guardan los seleccionados. Esta etapa no ejecuta segmentación, fine-tuning ni modelos deep learning. `smart_tissue_nuclei_v1` se conserva como versión intermedia/ablation para contrastar contra el baseline y contra v2_light.

## Legacy / experimental - smart_tissue_nuclei_v2_light

`smart_tissue_nuclei_v2_light` es una iteración previa del selector propio. Mejora v1 con proxy nuclear por HED color deconvolution, cuotas espaciales suaves por región y diversidad simple por features. Se conserva para trazabilidad/reproducibilidad, pero no es el método principal actual del paper.

```bash
conda run -n inf402-lumina-seg python scripts/06_select_wsi_patches.py \
  --wsi-path /Users/davidkripper/demoCasesMvpFeria/TCGA-A2-A3XS-01Z-00-DX1.867925C0-91D8-40A0-9FEA-25A635AC31E7.svs \
  --output-dir outputs/patch_selection/smart_v2_light_tcga_a2_a3xs \
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

Los outputs agregan trazabilidad de `nuclear_proxy`, región espacial, cuotas y `feature_diversity_bonus`. Las cuotas son suaves: evitan concentrar la selección en pocas zonas, pero no obligan a elegir patches de bajo score solo para llenar una región.

## Legacy / soporte interno - v3_server_quality

`v3_server_quality` es un selector pensado para ejecución en servidor/iHealth y se conserva como soporte interno/base técnica para v4.1. No usa deep learning ni el modelo de segmentación para seleccionar patches; mantiene la segmentación semántica como etapa posterior.

El selector usa proxies técnicos de utilidad esperada: calidad segmentable, señal nuclear HED/RGB, heterogeneidad, proxy prudente de baja celularidad compatible con lecho tratado, diversidad espacial y diversidad visual por features. Estos scores no son diagnóstico, no calculan RCB y no constituyen validación clínica.

Ejemplo de smoke test acotado:

```bash
conda run -n inf402-lumina-seg python scripts/06_select_wsi_patches.py \
  --wsi-path /Users/davidkripper/demoCasesMvpFeria/TCGA-A2-A3XS-01Z-00-DX1.867925C0-91D8-40A0-9FEA-25A635AC31E7.svs \
  --output-dir outputs/patch_selection/v3_server_quality_smoke \
  --selector v3_server_quality \
  --patch-size 1024 \
  --stride 1024 \
  --max-patches 4 \
  --min-tissue-ratio 0.20 \
  --seed 42 \
  --max-candidates-to-score 50 \
  --feature-size 256 \
  --quota-grid 2x2 \
  --overwrite
```

La salida sigue siendo compatible con `scripts/07_compare_patch_selectors.py`, `scripts/08_segment_selected_patches.py` y `scripts/09_compare_segmentation_on_selected_patches.py`.

## Legacy / experimental - v4_embedding_assisted

`v4_embedding_assisted` extiende `v3_server_quality` con embeddings UNI como representación morfológica. Los embeddings se usan para favorecer diversidad visual, balance de clusters y reducción de redundancia entre patches. UNI no se usa como clasificador clínico, no entrega ground truth y no reemplaza la segmentación posterior.

Esta etapa no usa `fcn_resnet50_unet-bcss` para seleccionar patches y no ejecuta segmentación preliminar. El modelo de segmentación sigue siendo una etapa posterior para generar máscaras/overlays revisables.

Requiere configurar acceso local a UNI mediante `--embedding-model-path` o reutilizar un cache de embeddings compatible. El script no descarga pesos automáticamente, no guarda tokens y no debe versionar pesos ni caches.

En el entorno local del repo se validó la integración y la falla limpia cuando UNI no está disponible; la ejecución real con UNI requiere entregar una ruta local válida mediante `--embedding-model-path` o un cache compatible.

Ejemplo de comando:

```bash
conda run -n inf402-lumina-seg python scripts/06_select_wsi_patches.py \
  --wsi-path /Users/davidkripper/demoCasesMvpFeria/TCGA-A2-A3XS-01Z-00-DX1.867925C0-91D8-40A0-9FEA-25A635AC31E7.svs \
  --output-dir outputs/patch_selection/v4_embedding_assisted_smoke \
  --selector v4_embedding_assisted \
  --patch-size 1024 \
  --stride 1024 \
  --max-patches 4 \
  --min-tissue-ratio 0.20 \
  --seed 42 \
  --max-candidates-to-score 50 \
  --feature-size 256 \
  --quota-grid 2x2 \
  --embedding-backend uni \
  --embedding-model-path /PATH/TO/UNI/MODEL_OR_CHECKPOINT \
  --embedding-device auto \
  --embedding-batch-size 8 \
  --cache-embeddings \
  --reuse-embedding-cache \
  --overwrite
```

Las salidas mantienen compatibilidad con las etapas posteriores y agregan `embedding_cache.npz`, `embedding_cache_metadata.json` y `embedding_cluster_summary.csv` cuando se calculan embeddings.

## Método principal propuesto - v4_1_medical_embedding_assisted

`v4_1_medical_embedding_assisted` es el selector técnico propuesto para la comparación principal del paper. Mantiene helpers/scoring base de `v3_server_quality`, agrega proxies clasicos de procesamiento de imagen medica y usa UNI solo como reranker morfologico dentro de candidatos tecnicamente fuertes.

Las features nuevas describen calidad tecnica de tincion/intensidad, mascara de tejido, textura, nitidez, artefactos y pseudo-celularidad basada en hematoxilina. Son proxies reproducibles para seleccion de patches; no son diagnostico, no segmentan nucleos reales, no calculan RCB y no constituyen validacion clinica.

La estrategia por defecto filtra candidatos por `score_v3_base`, calidad/utilidad medica y penalizacion de artefactos antes de usar embeddings UNI para diversidad morfologica controlada. La seleccion no usa `fcn_resnet50_unet-bcss` ni ejecuta segmentacion preliminar.

Ejemplo de comando con UNI local/cache compatible:

```bash
KMP_DUPLICATE_LIB_OK=TRUE /Users/davidkripper/miniforge3/envs/inf402-lumina-seg/bin/python scripts/06_select_wsi_patches.py \
  --wsi-path /Users/davidkripper/demoCasesMvpFeria/TCGA-A2-A3XS-01Z-00-DX1.867925C0-91D8-40A0-9FEA-25A635AC31E7.svs \
  --output-dir outputs/patch_selection/v4_1_medical_embedding_assisted_tcga_a2_a3xs \
  --selector v4_1_medical_embedding_assisted \
  --patch-size 1024 \
  --stride 1024 \
  --max-patches 16 \
  --min-tissue-ratio 0.20 \
  --seed 42 \
  --max-candidates-to-score 1000 \
  --feature-size 512 \
  --quota-grid 4x4 \
  --embedding-backend uni \
  --embedding-model-path /Users/davidkripper/models/uni/pytorch_model.bin \
  --embedding-device cpu \
  --embedding-batch-size 16 \
  --cache-embeddings \
  --reuse-embedding-cache \
  --medical-min-quality-score 0.50 \
  --medical-min-utility-score 0.45 \
  --min-score-v3-base-quantile 0.80 \
  --medical-top-quantile 0.20 \
  --medical-artifact-max 0.12 \
  --medical-rerank-mode top_v3_then_embedding \
  --overwrite
```

## Etapa 3 - comparación de selectores

La comparación formal toma dos carpetas ya generadas y recalcula features solo sobre los PNG seleccionados. La comparación principal actual es `baseline_tiatoolbox` vs `v4_1_medical_embedding_assisted`. Los pools iniciales pueden diferir: el baseline usa TIAToolbox/Otsu y v4.1 usa su generación propia con ranking técnico.

```bash
conda run -n inf402-lumina-seg python scripts/07_compare_patch_selectors.py \
  --baseline-dir outputs/patch_selection/baseline_tiatoolbox_tcga_a2_a3xs \
  --smart-dir outputs/patch_selection/v4_1_medical_embedding_assisted_tcga_a2_a3xs \
  --output-dir outputs/patch_selection/comparison_baseline_vs_v4_1_medical_embedding_assisted \
  --feature-size 256 \
  --overwrite
```

La salida incluye `comparison_summary.json`, `comparison_metrics.csv`, `selected_overlap.csv`, `comparison_selected_patches.csv`, `comparison_preview.png`, `comparison_preview_selected_only.png` y `comparison_notes.md`. Las métricas cubren conteos, overlap, features recomputadas, diversidad espacial y runtime. Esta etapa no ejecuta modelos ni valida desempeño clínico; para la visualización selected-only puede usar la WSI solo para reconstruir un thumbnail limpio si está disponible.

## Etapa 5 - segmentación sobre patches seleccionados

Después de seleccionar patches, se puede correr segmentación semántica técnica sobre los PNG ya guardados en `selected/`. Esta etapa reutiliza el flujo de inferencia existente y no repite selección de patches:

```bash
python scripts/08_segment_selected_patches.py \
  --input-selection-dir outputs/final_patch_selection/baseline_tiatoolbox \
  --output-dir outputs/segmentation/baseline_tiatoolbox \
  --model-name fcn_resnet50_unet-bcss \
  --device cpu \
  --input-mode patch \
  --limit-patches 2 \
  --overwrite
```

La salida incluye `per_patch/`, `masks/`, `overlays/`, `overlays_with_legend/`, `input_previews/`, `per_patch_segmentation.csv`, `inference_summary.json` y `method_config.json`. La máscara cruda del modelo puede tener menor resolución que el patch original; para el overlay se reescala con vecino más cercano para preservar etiquetas discretas de clase. Por eso, `class_pixel_counts` corresponde a la resolución cruda de predicción, no necesariamente al tamaño visual del overlay. Es segmentación técnica sobre patches seleccionados: no diagnostica, no calcula RCB, no reemplaza al patólogo y no constituye validación clínica.

## Etapa 5.5 - comparación de segmentación baseline vs v4.1

Una vez segmentados los patches seleccionados por cada método, se puede comparar técnicamente la distribución de clases predichas, warnings, tamaños de máscara y métricas operativas sin volver a ejecutar selección ni inferencia:

```bash
python scripts/09_compare_segmentation_on_selected_patches.py \
  --baseline-seg-dir outputs/segmentation/baseline_tiatoolbox \
  --smart-seg-dir outputs/segmentation/v4_1_medical_embedding_assisted \
  --output-dir outputs/segmentation/comparison_baseline_vs_v4_1_medical_embedding_assisted \
  --overwrite
```

La salida incluye `segmentation_comparison_summary.json`, métricas CSV, distribución de clases predichas, filas por patch, preview visual y notas Markdown. Esta comparación es técnica: no usa ground truth, no diagnostica, no calcula RCB y no constituye validación clínica.

## Prueba de carga del baseline TIAToolbox

Después de activar el ambiente reproducible, se puede ejecutar una prueba de carga del modelo preentrenado BCSS:

```bash
conda activate inf402-lumina-seg

python scripts/02_test_tiatoolbox_model.py \
  --model-name fcn_resnet50_unet-bcss \
  --device auto
```

El script intenta cargar el modelo preentrenado `fcn_resnet50_unet-bcss`, detecta PyTorch y el dispositivo disponible, y escribe un JSON de estado en `outputs/model_checks/tiatoolbox_bcss_model_status.json`.

Esta prueba solo valida carga del baseline. No ejecuta inferencia final, no diagnostica, no calcula RCB, no evalúa BCSS y no constituye validación clínica. Si TIAToolbox necesita descargar pesos para cargar el modelo, el cache debe quedar fuera del repositorio.

### Troubleshooting macOS: conflicto OpenMP/libomp

En macOS puede aparecer un error como:

```text
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program...
zsh: abort
```

Este error no necesariamente indica un problema del modelo ni del código de Lumina. Suele deberse a un conflicto entre librerías nativas usadas por dependencias como PyTorch, TIAToolbox, NumPy, scikit-image u otras que cargan runtime OpenMP.

Workaround probado para smoke tests locales en Mac:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python scripts/02_test_tiatoolbox_model.py \
  --model-name fcn_resnet50_unet-bcss \
  --device cpu
```

Opcionalmente, para reducir paralelismo:

```bash
OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE python scripts/02_test_tiatoolbox_model.py \
  --model-name fcn_resnet50_unet-bcss \
  --device cpu
```

`KMP_DUPLICATE_LIB_OK=TRUE` es un workaround temporal para pruebas locales, no una solución recomendada para experimentos finales, producción ni benchmarks formales. No lo exportes globalmente en el sistema sin entender las consecuencias.

En la prueba exitosa observada, TIAToolbox descargó pesos de aproximadamente `147 MB` y terminó con `Model loaded: OK`. Esos pesos deben quedar en cache fuera del repositorio y no deben subirse a Git. El estado de la prueba se guarda en `outputs/model_checks/tiatoolbox_bcss_model_status.json`.

## Inferencia smoke test con TIAToolbox

Después de validar que el baseline carga, se puede ejecutar una inferencia mínima sobre una imagen local pequeña para verificar que el modelo produce una salida visual. Para tiles o patches pequeños se debe usar `--input-mode patch`, que trata la imagen como un arreglo RGB en memoria y no exige metadata WSI como MPP u objetivo microscópico:

```bash
conda activate inf402-lumina-seg

KMP_DUPLICATE_LIB_OK=TRUE python scripts/04_run_inference.py \
  --image-path /Users/davidkripper/demoCasesMvpFeria/demo_case_01.tif \
  --model-name fcn_resnet50_unet-bcss \
  --device cpu \
  --input-mode patch \
  --output-dir outputs/inference_smoke/test_demo_case_01 \
  --clear-output
```

Esta prueba usa el flujo `SemanticSegmentor` de TIAToolbox y guarda un preview RGB, una máscara coloreada, un overlay, una leyenda visual y `inference_summary.json` en la carpeta de salida. Es solo un smoke test técnico: no evalúa calidad, no calcula métricas, no calcula RCB, no diagnostica y no constituye validación clínica.

El modelo entrega IDs numéricos de clase y los colores son asignados por el script de visualización. La salida incluye `legend.json` y `legend.png` con la relación `class_id -> color_rgb -> class_name/status -> pixel_count`. La paleta visual para demo usa `0 = Tumour` en rojo/crimson `(220, 20, 60)`, `1 = Stroma` en azul, `2 = Inflammatory` en verde, `3 = Necrosis` en naranja y `4 = Others` en morado. Los colores son solo visualización; el significado está dado por `class_id` y el mapping documentado.

### Importante: BCSS raw vs salida agrupada de TIAToolbox

No mezclar los códigos raw de BCSS con la salida agrupada del modelo preentrenado `fcn_resnet50_unet-bcss`.

En BCSS raw/original, las máscaras `.png` usan los códigos de `meta/gtruth_codes.tsv`; ahí `0 = outside_roi / don't care` y no significa `other`. Ese valor debe tratarse como región fuera de interés en el contexto de ground truth raw.

En la salida del modelo TIAToolbox `fcn_resnet50_unet-bcss`, la predicción está agrupada en cinco clases: `0 = Tumour`, `1 = Stroma`, `2 = Inflammatory`, `3 = Necrosis`, `4 = Others`. Por lo tanto, el valor `0` significa cosas distintas según el contexto: en BCSS raw es `outside_roi / don't care`, mientras que en la predicción agrupada TIAToolbox es `Tumour`.

`legend.json` y `inference_summary.json` registran `class_mapping_source`, `raw_bcss_zero_warning`, `bcss_raw_ground_truth_mapping` y `tiatoolbox_bcss_model_output_mapping` para evitar esta confusión. No usar el mapping raw para interpretar directamente la salida agrupada del modelo. Referencias: https://github.com/PathologyDataScience/BCSS, https://github.com/PathologyDataScience/BCSS/blob/master/meta/gtruth_codes.tsv y https://tia-toolbox.readthedocs.io/en/latest/_notebooks/jnb/06-semantic-segmentation.html.

Para WSI reales se puede usar `--input-mode wsi`, pero esas entradas normalmente necesitan metadata de escala o parámetros explícitos de lectura. Una TIFF pequeña sin MPP puede fallar en modo WSI con errores como `MPP is None`; en ese caso corresponde usar `--input-mode patch` mientras se trabaje con tiles locales. Si la predicción tiene un tamaño distinto al de la imagen de entrada, la máscara se redimensiona con vecino más cercano solo para construir el overlay. El JSON registra tanto el tamaño de predicción como el tamaño visualizado. Los outputs generados, leyendas, overlays y cualquier cache de pesos deben quedar fuera de Git.

## Advertencia sobre datos y pesos

No subir al repositorio:

- WSI o imágenes histopatológicas pesadas;
- datasets completos;
- datos clínicos sensibles;
- checkpoints o pesos de modelos;
- outputs generados;
- patches, máscaras, overlays o métricas derivados de datos grandes.

Usar `data/` y `outputs/` solo como estructura local de trabajo.

## Próximos hitos

1. Evaluar la comparación en más WSIs.
2. Incorporar ground truth BCSS cuando esté disponible para medir cobertura por clase.
3. Ejecutar segmentación semántica posterior sobre patches seleccionados para generar máscaras/overlays revisables.
4. Mantener `baseline_tiatoolbox` vs `v4_1_medical_embedding_assisted` como flujo principal del paper; conservar v1/v2/v3/v4 como legacy/soporte.
5. Considerar fine-tuning solo si la segmentación posterior no basta.
