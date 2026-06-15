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
- segmentación semántica;
- generación de máscaras y overlays;
- evaluación con métricas de segmentación cuando exista ground truth;
- posible fine-tuning si el baseline preentrenado no basta.

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
  -> patches útiles
  -> baseline preentrenado
  -> segmentación semántica
  -> máscara/overlay
  -> posible fine-tuning
```

La estrategia inicial es probar un baseline preentrenado, idealmente `fcn_resnet50_unet-bcss` documentado en TIAToolbox si se confirma disponibilidad local. Fine-tuning de U-Net/FPN/ResNet50-UNet queda como siguiente paso si el baseline no entrega resultados suficientes. Entrenar desde cero no es la primera opción.

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

1. Verificar ambiente local y Linux GPU.
2. Probar importación de TIAToolbox/OpenSlide.
3. Verificar disponibilidad operativa de `fcn_resnet50_unet-bcss` sin descargar pesos de forma implícita.
4. Ejecutar patching en imágenes pequeñas de prueba.
5. Preparar lectura y evaluación mínima con BCSS.
6. Evaluar baseline con overlays y métricas.
7. Considerar fine-tuning solo si el baseline no basta.
