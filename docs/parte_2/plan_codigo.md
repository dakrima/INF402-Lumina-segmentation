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

### 3. Patching inteligente

Implementar extracción de patches sobre imágenes pequeñas, guardar metadatos trazables y filtrar por proporción aproximada de tejido.

Estado inicial: el primer baseline de patching ya permite cortar imágenes pequeñas, calcular `tissue_ratio`, filtrar patches por umbral, guardar metadata CSV, generar un resumen JSON y producir un preview visual de la grilla seleccionada/rechazada. Esto sirve como base reproducible para documentar el flujo y extenderlo luego a WSI reales con OpenSlide/TIAToolbox.

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
