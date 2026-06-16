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

Estado de cierre para selección de patches: `baseline_tiatoolbox` queda como baseline comparativo, `smart_tissue_nuclei_v1` como versión intermedia/ablation y `smart_tissue_nuclei_v2_light` como selector propio candidato final del proyecto por ahora.

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
  -> baseline tipo TIAToolbox: grilla + máscara de tejido + min_mask_ratio
  -> selector propio de patches
  -> segmentación semántica posterior
  -> máscara/overlay revisable
  -> evaluación técnica y posible adaptación
```

La estrategia inicial de INF402 es formalizar un baseline de extracción tipo TIAToolbox y compararlo luego contra un selector propio de patches. La segmentación con `fcn_resnet50_unet-bcss` se mantiene como etapa posterior para generar máscaras/overlays revisables. Fine-tuning queda como opción posterior si la segmentación no es suficiente; entrenar desde cero no es la primera opción.

## Estructura del repositorio

```text
.
├── README.md
├── environment.yml
├── environment-linux-gpu.yml
├── requirements.txt
├── docs/
│   ├── parte_1/
│   └── parte_2/
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
│   └── figures/
├── notebooks/
├── scripts/
└── src/
```

`data/` y `outputs/` se mantienen con `.gitkeep`, pero su contenido real está ignorado por Git.

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

## Extracción básica de patches

El script de extracción actual está pensado para imágenes pequeñas (`.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`). Todavía no procesa WSI reales ni pirámides gigapixel; esa integración se hará después con TIAToolbox/OpenSlide.

Ejemplo:

```bash
python scripts/03_extract_patches.py \
  --image-path /ruta/imagen.tif \
  --patch-size 256 \
  --stride 256 \
  --min-tissue-ratio 0.99 \
  --output-dir outputs/patches/test_reconstructed \
  --clear-output \
  --save-rejected \
  --preview-image \
  --edge-policy overlap
```

`--min-tissue-ratio` define la fracción mínima aproximada de tejido para aceptar un patch. Se calcula con un umbral simple contra fondo blanco, por lo que sirve como baseline computacional inicial y no como decisión clínica.

`--edge-policy` controla qué ocurre cuando el tamaño de la imagen no calza exactamente con `patch_size` y `stride`:

- `drop`: ignora bordes incompletos. Es el comportamiento histórico.
- `overlap`: desplaza la última ventana para cubrir toda la imagen sin padding. Es la opción recomendada para experimentos iniciales cuando se quiere cobertura completa sin inventar píxeles.
- `pad`: cubre toda la imagen rellenando bordes con fondo blanco `(255, 255, 255)`.

Ejemplos:

```bash
# comportamiento actual
python scripts/03_extract_patches.py ... --edge-policy drop

# recomendado para cubrir toda la imagen sin padding
python scripts/03_extract_patches.py ... --edge-policy overlap

# útil cuando se quiere cubrir todo incluso con padding
python scripts/03_extract_patches.py ... --edge-policy pad
```

`--clear-output` limpia la carpeta indicada por `--output-dir` antes de correr, con restricciones de seguridad para no borrar `/`, home, la raíz del repo ni carpetas peligrosas. Úsalo solo cuando quieras regenerar una corrida.

La salida incluye:

- `selected/`: patches aceptados;
- `rejected/`: patches rechazados, solo si se usa `--save-rejected`;
- `patches_metadata.csv`: todos los patches evaluados, incluyendo coordenadas, `tissue_ratio`, `selected`, `saved`, `split`, `edge_policy`, `padded` y dimensiones originales;
- `summary.json`: resumen de la corrida, incluyendo cobertura de imagen y cantidad de patches con padding;
- `patch_selection_preview.png`: grilla visual sobre la imagen original si se usa `--preview-image`.

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
python scripts/06_select_wsi_patches.py \
  --wsi-path /Users/davidkripper/demoCasesMvpFeria/TCGA-A2-A3XS-01Z-00-DX1.867925C0-91D8-40A0-9FEA-25A635AC31E7.svs \
  --output-dir outputs/patch_selection/baseline_tcga_a2_a3xs \
  --selector baseline_tiatoolbox \
  --patch-size 1024 \
  --stride 1024 \
  --max-patches 16 \
  --min-tissue-ratio 0.20 \
  --seed 42 \
  --overwrite
```

La salida incluye `selected/`, `candidate_metadata.csv`, `selected_metadata.csv`, `selection_summary.json`, `method_config.json` y `patch_selection_preview.png`. `candidate_metadata.csv` contiene el pool común de candidatos filtrados por thumbnail, mientras que `selected_metadata.csv` contiene solo los patches seleccionados.

Limitación actual: este baseline tipo TIAToolbox no usa ranking inteligente, señal nuclear, diversidad espacial, HoVer-Net, CLAM ni active learning. La separación entre pool de candidatos y seleccionados permite una comparación justa contra `smart_tissue_nuclei_v1` y `smart_tissue_nuclei_v2_light` bajo el mismo presupuesto de patches.

## Etapa 2 - smart_tissue_nuclei_v1

El selector propio `smart_tissue_nuclei_v1` usa el mismo pool común de candidatos filtrados por thumbnail y scorea una muestra reproducible de candidatos con features simples: `tissue_ratio`, `nuclear_signal`, `visual_entropy`, `blur_score`, `artifact_penalty` y penalización espacial greedy.

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

## Etapa 2.1 - smart_tissue_nuclei_v2_light

`smart_tissue_nuclei_v2_light` mejora el selector propio con tres mecanismos livianos: proxy nuclear por HED color deconvolution, cuotas espaciales suaves por región y diversidad simple por features. Mantiene el mismo pool thumbnail-filtered, no usa modelos deep learning y queda congelado por ahora como selector propio candidato final.

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

## Etapa 3 - comparación de selectores

La comparación formal toma dos carpetas ya generadas, valida que compartan configuración experimental y recalcula features solo sobre los PNG seleccionados. La comparación principal de cierre es `baseline_tiatoolbox` vs `smart_tissue_nuclei_v2_light`:

```bash
conda run -n inf402-lumina-seg python scripts/07_compare_patch_selectors.py \
  --baseline-dir outputs/patch_selection/baseline_tcga_a2_a3xs \
  --smart-dir outputs/patch_selection/smart_v2_light_tcga_a2_a3xs \
  --output-dir outputs/patch_selection/comparison_baseline_vs_smart_v2_light_tcga_a2_a3xs \
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

## Etapa 5.5 - comparación de segmentación baseline vs smart_v2

Una vez segmentados los patches seleccionados por cada método, se puede comparar técnicamente la distribución de clases predichas, warnings, tamaños de máscara y métricas operativas sin volver a ejecutar selección ni inferencia:

```bash
python scripts/09_compare_segmentation_on_selected_patches.py \
  --baseline-seg-dir outputs/segmentation/baseline_tiatoolbox \
  --smart-seg-dir outputs/segmentation/smart_tissue_nuclei_v2_light \
  --output-dir outputs/segmentation/comparison_baseline_vs_smart_v2_light \
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
4. Mantener `smart_tissue_nuclei_v2_light` como selector candidato final mientras no exista evidencia experimental para cambiar pesos o features.
5. Considerar fine-tuning solo si la segmentación posterior no basta.
