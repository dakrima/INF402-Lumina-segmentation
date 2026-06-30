# INF402 - Lumina Segmentation

Repositorio del experimento presentado en el paper **“Selección de patches en whole-slide images histopatológicas: comparación entre TIAToolbox y un método asistido por embeddings morfológicos”**.

## Objetivo

Comparar un baseline TIAToolbox con un selector que combina calidad técnica, distribución espacial y embeddings UNI para elegir patches informativos desde un pool común.

## Alcance técnico

El experimento incluye:

- generación del pool común con TIAToolbox y máscara Otsu;
- selección baseline reproducible con semilla fija;
- extracción de features técnicas y médicas clásicas;
- embeddings UNI, clustering y reranking morfológico;
- criterios de diversidad espacial;
- métricas técnicas, espaciales, morfológicas y de tiempo.

El flujo compara `baseline_tiatoolbox` con `v4_1_medical_embedding_assisted` sobre nueve WSI, con 16 patches por WSI y método. La segmentación downstream no forma parte de este experimento.

## Flujo del experimento

```text
WSI / imagen histopatológica H&E
  -> pool común TIAToolbox: ventana deslizante + máscara Otsu + min_mask_ratio
  -> baseline reproducible
  -> selector propuesto: features + UNI + clustering + reranking
  -> métricas por WSI
  -> tablas agregadas
```

Ambos métodos reciben el mismo pool inicial y el mismo presupuesto. `baseline_tiatoolbox` selecciona de forma reproducible desde ese pool; `v4_1_medical_embedding_assisted` aplica scoring técnico, proxies de imagen médica, embeddings UNI y reranking morfológico.

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

Las WSI, los pesos UNI y las corridas completas se mantienen fuera de Git. Solo se versionan los agregados utilizados para verificar las cifras del paper.

## Instalación

Conda/Mamba permite coordinar las dependencias Python y librerías nativas como OpenSlide entre macOS, Linux y servidores con GPU.

```bash
mamba env create -f environment.yml
mamba activate inf402-lumina-seg
```

Si `mamba` no está disponible, se puede usar `conda env create -f environment.yml`.

## Verificación del entorno

```bash
python scripts/verificar_entorno.py
```

El script revisa las dependencias importadas por el pipeline y las carpetas mínimas.

## Ejecución del experimento

La ubicación esperada de las nueve WSI se describe en `data/README.md`, y la del checkpoint UNI en `models/README.md`. Con el ambiente activo:

```bash
python scripts/ejecutar_experimento.py
python scripts/generar_resultados.py
```

Los parámetros metodológicos del paper están fijados en `scripts/ejecutar_experimento.py`; los argumentos de terminal configuran rutas, cantidad esperada de casos, sobrescritura y autocomprobaciones.

## Resultados

Los resultados agregados utilizados en el paper se encuentran en:

- `results/metrics/`
- `results/tables/`

Las corridas completas se generan bajo `results/runs/` y no se versionan debido a su tamaño.
