# Plan de código - Parte II INF402

## Estrategia de desarrollo

El desarrollo partirá con una base reproducible y verificable antes de ejecutar inferencia pesada. La prioridad es controlar ambiente, rutas, patching y evaluación mínima antes de descargar datasets o pesos.

La metodología se mantiene conservadora:

1. validar ambiente;
2. probar herramientas de lectura e inferencia;
3. ejecutar patching en imágenes pequeñas;
4. usar BCSS como dataset técnico de segmentación semántica;
5. evaluar baseline;
6. hacer fine-tuning solo si el baseline no basta.

## Hitos

### 1. Repo y ambiente

Crear estructura del repositorio, ambientes Conda/Mamba, `requirements.txt`, `.gitignore`, rutas centralizadas y script de verificación.

### 2. Prueba TIAToolbox

Confirmar importación de TIAToolbox/OpenSlide y disponibilidad operativa del modelo objetivo `fcn_resnet50_unet-bcss`. No descargar pesos grandes automáticamente.

Comando de smoke test:

```bash
conda activate inf402-lumina-seg

python scripts/02_test_tiatoolbox_model.py \
  --model-name fcn_resnet50_unet-bcss \
  --device auto
```

Esta verificación solo confirma carga del baseline preentrenado y genera `outputs/model_checks/tiatoolbox_bcss_model_status.json`. No hace inferencia clínica, no evalúa BCSS, no calcula RCB y no entrena modelos.

#### Troubleshooting macOS: conflicto OpenMP/libomp

En macOS, el smoke test puede abortar con:

```text
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
```

Esto no implica necesariamente que el modelo `fcn_resnet50_unet-bcss` o el código de Lumina estén malos. Normalmente apunta a un conflicto entre runtimes OpenMP cargados por dependencias nativas como PyTorch, TIAToolbox, NumPy, scikit-image u otras.

Workaround probado para una verificación local:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python scripts/02_test_tiatoolbox_model.py \
  --model-name fcn_resnet50_unet-bcss \
  --device cpu
```

Si se quiere reducir paralelismo durante esta prueba:

```bash
OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE python scripts/02_test_tiatoolbox_model.py \
  --model-name fcn_resnet50_unet-bcss \
  --device cpu
```

`KMP_DUPLICATE_LIB_OK=TRUE` debe tratarse como workaround temporal de smoke test local. No usarlo como configuración global sin revisar consecuencias, ni considerarlo solución para experimentos finales, producción o benchmarks. En la corrida exitosa observada se descargaron pesos de aproximadamente `147 MB`; deben quedar cacheados fuera del repositorio y no subirse a Git. El resultado esperado es `Model loaded: OK` y el JSON queda en `outputs/model_checks/tiatoolbox_bcss_model_status.json`.

#### Inferencia smoke test local

Una vez confirmada la carga del baseline, el siguiente smoke test técnico verifica que `fcn_resnet50_unet-bcss` puede generar una salida visual sobre una imagen local pequeña. Para tiles o patches locales, usar `--input-mode patch`:

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

El comando genera `input_preview.png`, `prediction_mask.png`, `prediction_overlay.png`, `legend.json` e `inference_summary.json` bajo `outputs/inference_smoke/test_demo_case_01/`. Esta corrida no evalúa calidad, no calcula Dice/IoU, no calcula RCB, no diagnostica y no valida clínicamente el sistema; solo comprueba que el baseline puede producir una máscara/overlay revisable.

El script registra en el JSON el modo de entrada, `patch_mode`, tamaño de la imagen, dispositivo usado, versiones de TIAToolbox/PyTorch, clases de modelo/configuración, forma de la predicción, etiquetas observadas, conteos por clase y rutas de salida. Si el tamaño de salida no coincide con el input, el resize se usa únicamente para visualización del overlay.

El modelo devuelve IDs numéricos de clase. Los colores de `prediction_mask.png` y `prediction_overlay.png` son asignados por el script, no necesariamente por el modelo. `legend.json` documenta `class_id -> color_rgb -> class_name/status -> pixel_count`; los nombres solo deben interpretarse si `mapping_source` aparece confirmado desde TIAToolbox/BCSS. Mientras el mapping esté como `unconfirmed`, la máscara sigue siendo una visualización técnica sin interpretación clínica.

#### Importante: BCSS raw vs salida agrupada de TIAToolbox

El mapping raw/original de BCSS y el mapping de salida del modelo `fcn_resnet50_unet-bcss` son distintos y no deben mezclarse.

En BCSS raw/original, los códigos de ground truth están documentados en `meta/gtruth_codes.tsv`: `0 = outside_roi / don't care`. Ese `0` no significa `other` y corresponde a regiones fuera del ROI en las máscaras raw.

En el modelo preentrenado de TIAToolbox, la salida está agrupada en cinco clases: `0 = Tumour`, `1 = Stroma`, `2 = Inflammatory`, `3 = Necrosis`, `4 = Others`. Por eso, el `0` de una predicción agrupada TIAToolbox debe leerse como `Tumour`, no como `outside_roi`.

El smoke test registra ambos contextos en `legend.json` e `inference_summary.json` mediante `bcss_raw_ground_truth_mapping`, `tiatoolbox_bcss_model_output_mapping`, `class_mapping_source` y `raw_bcss_zero_warning`. No usar la salida del smoke test para diagnóstico clínico, cálculo de RCB ni validación clínica.

Referencias: https://github.com/PathologyDataScience/BCSS, https://github.com/PathologyDataScience/BCSS/blob/master/meta/gtruth_codes.tsv y https://tia-toolbox.readthedocs.io/en/latest/_notebooks/jnb/06-semantic-segmentation.html.

Para WSI reales se reserva `--input-mode wsi`. Ese modo puede requerir metadata de escala, como MPP u objective power, o parámetros explícitos de lectura. Una TIFF pequeña sin esa metadata no debe tratarse como WSI en este hito porque puede fallar con errores de escala; para esos casos corresponde `--input-mode patch`.

### 3. Patching inteligente

Implementar extracción de patches sobre imágenes pequeñas, guardar metadatos trazables y filtrar por proporción aproximada de tejido.

Estado inicial: el primer baseline de patching ya permite cortar imágenes pequeñas, calcular `tissue_ratio`, filtrar patches por umbral, guardar metadata CSV, generar un resumen JSON y producir un preview visual de la grilla seleccionada/rechazada. Esto sirve como base reproducible para documentar el flujo y extenderlo luego a WSI reales con OpenSlide/TIAToolbox.

El patching de imágenes pequeñas ahora soporta políticas de borde: `drop`, `overlap` y `pad`. Para experimentos iniciales se recomienda `overlap` cuando se necesita cubrir toda la imagen sin inventar píxeles; en WSI reales esta lógica deberá adaptarse a lectura por tiles con OpenSlide/TIAToolbox.

### 4. BCSS mínimo

Incorporar BCSS como dataset principal de segmentación semántica cuando se definan rutas, permisos y formato de descarga. No se debe subir BCSS al repositorio.

### 5. Evaluación baseline

Comparar predicciones contra ground truth cuando exista, usando pixel accuracy, IoU/mIoU y Dice si se incorpora. Complementar con revisión visual de overlays.

### 6. Fine-tuning si hace falta

Si el baseline preentrenado no alcanza desempeño suficiente, evaluar fine-tuning de U-Net/FPN/ResNet50-UNet sobre BCSS o datos objetivo anotados.

## Rol de BCSS

BCSS será usado como:

- dataset principal para segmentación semántica;
- fuente de clases regionales y ground truth;
- base para evaluación cuantitativa;
- posible base de fine-tuning.

BCSS no debe presentarse como validación clínica post-neoadyuvancia. Su rol es técnico y metodológico, no clínico definitivo.

## Rol de servidores iHealth

Los servidores iHealth con A100/H100 permiten fine-tuning, comparación de modelos y experimentos más pesados. Aun así, la estrategia no cambia: primero se valida un baseline preentrenado, luego se mide y solo después se escala.

El acceso a cómputo reduce restricciones de entrenamiento, pero no reemplaza una metodología reproducible.

## Pendientes operativos

- Confirmar instalación de OpenSlide en macOS y Linux.
- Confirmar instalación de TIAToolbox.
- Verificar compatibilidad real de `fcn_resnet50_unet-bcss`.
- Validar combinación PyTorch/CUDA en iHealth o NLHPC.
- Definir clases finales del PMV.
- Definir formato de salida requerido por el grupo de cuantificación.
- Definir política local para datasets, WSI, checkpoints y outputs.
