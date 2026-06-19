# Plan para inferencia con contexto y stitching

## 1. Objetivo

Preparar una validacion inicial para la estrategia `context-stitch-2x2` sobre patches seleccionados. Esta etapa no integra el metodo al batch productivo, no cambia el modelo y no reemplaza la inferencia `single-window`.

Advertencia obligatoria: Technical segmentation/inference only. Not for diagnosis, not RCB, not clinical validation.

## 2. Validacion inicial: geometria y contexto

La primera etapa separa tres preguntas tecnicas:

- Si la geometria 2x2 reconstruye correctamente una mascara de `1024x1024` desde cuatro salidas simuladas de `512x512`.
- Si se puede leer desde la WSI original una region de contexto `1536x1536` alrededor de un patch seleccionado.
- Si una prueba minima con modelo produce cuatro salidas `512x512` que pueden stitched en una mascara `1024x1024`.

La prueba sintetica valida solamente geometria y convencion de ejes. No demuestra por si sola que la salida `512x512` del modelo corresponda a la zona central util del input `1024x1024`.

## 3. Geometria propuesta

Para `patch_input_shape=1024` y `patch_output_shape=512`, el margen tecnico es:

```text
margin = (1024 - 512) / 2 = 256
context_size = 1024 + 2 * 256 = 1536
```

Las coordenadas WSI/PIL/OpenSlide usan `(x, y)`. Los arrays NumPy usan `[y, x]`.

Ventanas:

- `window_00`: input `y=0:1024, x=0:1024`, target `y=0:512, x=0:512`.
- `window_01`: input `y=0:1024, x=512:1536`, target `y=0:512, x=512:1024`.
- `window_10`: input `y=512:1536, x=0:1024`, target `y=512:1024, x=0:512`.
- `window_11`: input `y=512:1536, x=512:1536`, target `y=512:1024, x=512:1024`.

## 4. Lectura desde WSI

El metodo debe usar `source_wsi_path` o `wsi_path`, junto con `x_level0`, `y_level0` y `patch_size`. No debe depender solo del PNG seleccionado, porque los bordes del patch objetivo requieren contexto externo.

Si el contexto solicitado cae fuera de la WSI, la etapa inicial usa padding blanco (`RGB 255,255,255`) y lo registra en el manifest:

- `context_padding_used`
- `padding_left`
- `padding_right`
- `padding_top`
- `padding_bottom`
- `padding_mode`

## 5. Prueba minima con modelo

El probe opcional `--run-alignment-probe` corre `fcn_resnet50_unet-bcss` sobre las cuatro ventanas `1024x1024`, carga cada `prediction_labels_raw.npy`, verifica salidas `512x512` y genera:

- `stitched_prediction_1024.npy`
- `stitched_prediction_1024.png`
- `stitched_overlay_1024.png`
- `alignment_probe_manifest.json`

El manifest debe mantener `hypothesis_confirmed = "visual_review_required"`. Esto indica que la geometria y los shapes son consistentes, pero la hipotesis del centro util requiere revision visual o comparacion formal contra inferencia WSI/region de TIAToolbox.

## 6. Limites

Esta etapa no implementa QC avanzado, no agrega agregacion WSI final, no calcula RCB, no valida clinicamente y no reemplaza revision experta. Solo prepara trazabilidad tecnica para decidir si `context-stitch-2x2` merece integrarse despues como estrategia opcional.

## 7. Verificacion de la ubicacion espacial del output

Antes de integrar `context-stitch-2x2` al batch productivo, se agrega un probe separado:

```bash
python scripts/11_probe_tiatoolbox_output_placement.py \
  --selection-dir outputs/patch_selection/v4_1_medical_embedding_assisted_tcga_a2_a3xs \
  --patch-index 0 \
  --output-dir outputs/context_stitch_probe/output_placement \
  --run-source-inspection \
  --run-coordinate-probe \
  --run-tiatoolbox-merge-probe \
  --overwrite
```

Este probe inspecciona la configuracion y el codigo instalado de TIAToolbox para responder si la salida `512x512` del modelo se ubica como prediccion central dentro del input `1024x1024`. La evidencia primaria viene de:

- `UNetModel.infer_batch`, que aplica `centre_crop` despues de interpolar la prediccion.
- `PatchExtractor.get_coordinates`, que centra la ventana de entrada grande alrededor del bound de salida mas pequeno.
- `WSIPatchDataset` y `SemanticSegmentor`, que transportan `output_locs` separados de las coordenadas de entrada.

La comparacion directa contra una mascara stitched puede ejecutarse como chequeo secundario, pero no debe tratarse como prueba definitiva. El resultado esperado del probe debe clasificarse explicitamente como `supported`, `contradicted` o `inconclusive`, y siempre bajo la advertencia: Technical segmentation/inference only. Not for diagnosis, not RCB, not clinical validation.

## 8. Comparacion 2x2 sin overlap vs overlap-aware

La siguiente etapa experimental compara dos formas de reconstruir una mascara tecnica `1024x1024` sobre el mismo patch objetivo:

- `context-stitch-2x2` sin overlap: cuatro ventanas `1024x1024`, cuatro salidas centrales `512x512` y stitching directo por cuadrantes.
- `overlap-aware`: ventanas definidas con la logica de coordenadas de TIAToolbox y `stride_shape=(450, 450)`, acumulando probabilidades en el canvas objetivo antes de aplicar `argmax`.

Comando base:

```bash
KMP_DUPLICATE_LIB_OK=TRUE \
NUMBA_CACHE_DIR=/tmp/numba_cache \
MPLCONFIGDIR=/tmp/mpl_config \
/Users/davidkripper/miniforge3/envs/inf402-lumina-seg/bin/python \
  scripts/12_compare_context_stitch_strategies.py \
  --selection-dir outputs/patch_selection/v4_1_medical_embedding_assisted_tcga_a2_a3xs \
  --patch-indices 0,1,2 \
  --output-dir outputs/context_stitch_comparison \
  --model-name fcn_resnet50_unet-bcss \
  --device cpu \
  --overlap-stride 450 \
  --blend-mode uniform \
  --run-no-overlap \
  --run-overlap-aware \
  --overwrite
```

La mezcla overlap-aware usa probabilidades, no IDs de clase interpolados. El modo principal es `uniform`; el modo `feathered` queda disponible como variante tecnica con pesos suaves y epsilon minimo para evitar pixeles sin cobertura.

Metricas registradas:

- rendimiento: cantidad de ventanas, runtime total y runtime promedio por ventana;
- distribucion de clases predichas por estrategia;
- acuerdo entre estrategias: pixel agreement, disagreement ratio, IoU entre estrategias y diferencias de ratios de clase;
- continuidad tecnica en uniones: discontinuidad de labels, probabilidades y confianza;
- cobertura overlap-aware: conteo de cobertura, pesos acumulados y pixeles sin cobertura.

Estas metricas comparan estrategias de inferencia, no exactitud contra ground truth. No permiten afirmar que una estrategia diagnostica mejor, estima RCB ni mejora desempeno clinico. La recomendacion final debe limitarse a `prefer_no_overlap`, `prefer_overlap_aware`, `technically_similar` o `inconclusive`, con razonamiento tecnico y sin claims clinicos.
