# INF402 - Lumina Segmentation

Repositorio del experimento presentado en el paper **“Selección de patches en whole-slide images histopatológicas: comparación entre TIAToolbox y un método asistido por embeddings morfológicos”**.

## Objetivo

El proyecto compara un baseline TIAToolbox con un selector que combina calidad técnica, distribución espacial y embeddings UNI para elegir patches informativos desde un pool común.

La salida del grupo es un insumo técnico para análisis posterior. El sistema debe entenderse como apoyo académico/computacional y no como herramienta clínica autónoma.

## Alcance técnico

El alcance conservado es:

- generación del pool común con TIAToolbox y máscara Otsu;
- selección baseline reproducible con semilla fija;
- extracción de features técnicas y médicas clásicas;
- embeddings UNI, clustering y reranking morfológico;
- criterios de diversidad espacial;
- métricas técnicas, espaciales, morfológicas y de tiempo.

El flujo compara `baseline_tiatoolbox` contra `v4_1_medical_embedding_assisted` sobre nueve WSI, con 16 patches por WSI y método.

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
  -> pool común TIAToolbox: ventana deslizante + máscara Otsu + min_mask_ratio
  -> baseline reproducible
  -> selector propuesto: features + UNI + clustering + reranking
  -> métricas por WSI
  -> tablas agregadas
```

Ambos métodos reciben exactamente el mismo pool y el mismo presupuesto. La segmentación downstream no forma parte del experimento conservado en este repositorio.

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

Para aislar el efecto del ranking y selección final, `baseline_tiatoolbox` y `v4_1_medical_embedding_assisted` parten del mismo pool inicial de candidatos generado con TIAToolbox `SlidingWindowPatchExtractor`, `input_mask="otsu"` y `min_mask_ratio`. El baseline selecciona de forma reproducible desde ese pool, mientras que v4.1 aplica scoring técnico, proxies de imagen médica, embeddings UNI y reranking morfológico sobre el mismo universo inicial.

## Estructura del repositorio

```text
.
├── README.md
├── environment.yml
├── data/
│   └── README.md
├── models/
│   └── README.md
├── results/
│   ├── metrics/
│   └── tables/
├── scripts/
│   ├── ejecutar_experimento.py
│   ├── generar_resultados.py
│   └── verificar_entorno.py
└── src/
```

Las WSI, los pesos UNI y las corridas pesadas se mantienen fuera de Git. Solo se versionan los agregados pequeños utilizados para verificar las cifras del paper.

## Ambientes reproducibles

Se usa Conda/Mamba como primera opción porque permite coordinar dependencias Python y librerías nativas como OpenSlide de forma más controlada entre macOS, Linux y servidores con GPU.

## Instalación local

```bash
mamba env create -f environment.yml
mamba activate inf402-lumina-seg
python scripts/verificar_entorno.py
```

Si no tienes `mamba`, puedes usar `conda env create -f environment.yml`.

## Verificación de ambiente

```bash
python scripts/verificar_entorno.py
```

El script revisa las dependencias efectivamente importadas por el pipeline y las carpetas mínimas.

## Ejecución provisional del flujo final

Con el ambiente activo, ubicar las nueve WSI según `data/README.md` y el checkpoint UNI según `models/README.md`. Luego ejecutar:

```bash
python scripts/verificar_entorno.py
python scripts/ejecutar_experimento.py
python scripts/generar_resultados.py
```

Los parámetros metodológicos del paper están fijados en `ejecutar_experimento.py`; los argumentos de terminal solo configuran rutas, cantidad esperada de casos, sobrescritura y autocomprobaciones.

## Pendientes para la reescritura del README

- documentar la fuente exacta y condiciones de acceso de las nueve WSI;
- resolver la discrepancia entre la mención a BACH y los identificadores `TCGA-*` preservados;
- explicar la obtención autorizada del checkpoint UNI;
- describir cada archivo versionado en `results/metrics/` y `results/tables/`;
- agregar tiempos y requisitos de hardware observados;
- incorporar una tabla que relacione métricas y tablas con el manuscrito;
- registrar las validaciones completas cuando se repita el experimento desde cero.
