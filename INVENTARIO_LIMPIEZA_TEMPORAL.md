# Inventario previo a la limpieza del repositorio

Este inventario se registró antes de eliminar archivos. La clasificación se obtuvo revisando
la estructura versionada, los imports de todos los archivos Python, las referencias textuales,
el historial reciente y los resultados locales de la corrida `inf402_n9`.

El flujo que produjo los resultados finales está compuesto por:

1. `scripts/13_run_inf402_patch_selection_experiment.py`, que genera una vez el pool común con
   TIAToolbox, ejecuta el baseline y el selector v4.1, compara ambos métodos y agrega métricas.
2. `scripts/14_analyze_morphological_diversity.py`, que reutiliza los embeddings UNI persistidos
   para calcular las métricas morfológicas finales.

## Conservar sin cambios estructurales

- `LICENSE`: licencia del repositorio; no interviene en la ejecución.
- `data/`: punto de entrada para las WSI externas. Se reemplazarán los `.gitkeep` redundantes por
  una instrucción breve porque BACH no debe versionarse.
- `src/selection/comparison.py`: calcula las métricas técnicas y espaciales usadas por el script
  final. Lo importa directamente el experimento.
- `src/selection/diversity.py`: contiene distancias, cuotas espaciales y diversidad de features.
  Lo usa directamente el selector propuesto.
- `src/selection/embedding_scoring.py`: carga UNI, genera/reutiliza embeddings, hace clustering y
  calcula distancias coseno. Lo usan el selector y el análisis morfológico.
- `src/selection/manifests.py`: define y escribe los CSV/JSON consumidos por todo el flujo final.
- `src/selection/medical_image_features.py`: extrae los proxies clásicos de imagen médica del
  método propuesto.
- `src/selection/previews.py` y `src/visualization/patch_preview.py`: generan las previews del
  baseline y del selector propuesto.
- `src/selection/quality_filters.py`: calcula tejido, HED, entropía, nitidez y artefactos; también
  se usa al recomputar métricas para la comparación.
- `src/selection/tiatoolbox_baseline.py`: genera el pool común TIAToolbox/Otsu y ejecuta el
  baseline reproducible.
- `src/preprocessing/wsi_patch_extraction.py`: aporta lectura OpenSlide, metadata, limpieza segura
  de salidas y máscara auxiliar usada por las features.

## Conservar, pero refactorizar

- `scripts/13_run_inf402_patch_selection_experiment.py`: es el ejecutor final, pero tiene nombre
  numerado, rutas absolutas personales, mensajes en inglés y defaults no portables. Se renombrará
  a `scripts/ejecutar_experimento.py`, manteniendo exactamente sus parámetros metodológicos.
- `scripts/14_analyze_morphological_diversity.py`: es la segunda etapa final; se renombrará a
  `scripts/generar_resultados.py` y se conservarán sus fórmulas y validaciones.
- `scripts/01_check_environment.py`: se reducirá a las dependencias y carpetas del experimento
  final, y se renombrará a `scripts/verificar_entorno.py`.
- `src/selection/v4_1_medical_embedding_assisted.py`: contiene el método propuesto definitivo,
  pero su nombre refleja una versión y depende de helpers privados alojados en implementaciones
  antiguas completas. Se renombrará de forma descriptiva y se aislarán solo esos helpers.
- `src/selection/v3_server_quality.py`: no se conservará como selector ejecutable. Únicamente sus
  funciones de scoring técnico usadas por v4.1 pasarán a un módulo descriptivo compartido.
- `src/selection/v4_embedding_assisted.py`: no se conservará como selector ejecutable. Únicamente
  sus helpers de cache, embeddings y resumen de clusters usados por v4.1 pasarán a un módulo
  descriptivo compartido.
- `src/selection/candidate_generation.py`: el flujo final solo necesita la estructura
  `PatchCandidate`; se eliminará la generación thumbnail de pipelines anteriores.
- `src/selection/scoring.py`: el flujo final solo necesita `normalize_feature`; se reducirá a esa
  normalización usada por el scoring técnico definitivo.
- `src/selection/__init__.py`: actualmente exporta cinco selectores; quedará limitado al baseline
  y al método propuesto.
- `environment.yml`: contiene herramientas y librerías de segmentación/notebooks no usadas y no
  declara explícitamente `timm`, `scikit-learn` ni `psutil`. Quedará como única definición de
  ambiente para el experimento final.
- `.gitignore`: se ajustará a `results/`, `data/` y `models/` sin conservar árboles vacíos de
  outputs descartados.
- `README.md`: no se reescribirá por completo. Solo se corregirán rutas/comandos rotos y se
  registrarán las secciones pendientes para una tarea posterior.

## Mover o reorganizar

- Los dos scripts finales numerados se moverán a nombres descriptivos en `scripts/`.
- El selector `v4_1_medical_embedding_assisted.py` se moverá a un nombre estable que describa el
  método propuesto sin codificar una versión histórica.
- Los resultados agregados pequeños de `outputs/patch_selection/inf402_n9/aggregate/` se
  conservarán en `results/metrics/` y `results/tables/`; se excluirán caches, patches PNG y salidas
  intermedias. Antes de moverlos se comprobó que fueron generados por los scripts 13 y 14, que
  corresponden a nueve WSI y que los archivos contienen resultados completos.
- `outputs/` dejará de ser la raíz por defecto; las nuevas corridas usarán `results/runs/`, sin
  cambiar nombres de archivos, columnas, claves ni cálculos del experimento.

## Eliminar

### Código antiguo y pruebas exploratorias

- `archive/legacy_small_image_pipeline/`: pipeline completo para imágenes pequeñas, notebook y
  extracción previa a WSI. Ningún import del flujo final apunta a `archive/`; la comprobación fue
  el grafo AST de imports y una búsqueda global por sus rutas.
- `scripts/05_extract_wsi_patches.py`: extracción OpenSlide independiente reemplazada por la
  generación común TIAToolbox del experimento final. No es importado; se comprobó por búsqueda de
  imports y referencias.
- `scripts/06_select_wsi_patches.py`: CLI multipropósito para cinco selectores históricos. El
  paper usa el ejecutor de nueve WSI, que llama directamente al baseline y v4.1. No es importado
  por el flujo final.
- `scripts/07_compare_patch_selectors.py`: CLI manual reemplazada por la comparación integrada en
  el ejecutor final. `src/selection/comparison.py` sí se conserva.
- `src/selection/smart_tissue_nuclei.py`: implementaciones v1/v2 descartadas. Solo se exporta desde
  `src/selection/__init__.py`; no la usa el flujo final.
- `src/selection/v3_server_quality.py`: selector v3 descartado. Tras extraer sus helpers usados por
  el método final, no quedarán referencias vigentes.
- `src/selection/v4_embedding_assisted.py`: selector v4 descartado. Tras extraer sus helpers usados
  por el método final, no quedarán referencias vigentes.

### Segmentación downstream no incluida en el paper

- `scripts/02_test_tiatoolbox_model.py`, `scripts/04_run_inference.py`,
  `scripts/05_evaluate_bcss.py`, `scripts/08_segment_selected_patches.py` y
  `scripts/09_compare_segmentation_on_selected_patches.py`: pruebas/corridas del modelo BCSS y
  segmentación posterior. Ninguna participa en la selección ni en las métricas finales.
- `src/inference/`, `src/models/`, `src/evaluation/`, `src/io/` y
  `src/visualization/overlays.py`, `src/visualization/segmentation_overlay.py`: soporte exclusivo
  de inferencia/segmentación/overlays. El grafo de imports de los scripts 13 y 14 no llega a estos
  módulos.

### Context stitching, probes y geometría descartada

- `scripts/10_probe_context_stitch_geometry.py`,
  `scripts/11_probe_tiatoolbox_output_placement.py` y
  `scripts/12_compare_context_stitch_strategies.py`: probes de una etapa posterior no presentada.
  Solo importan `src/inference/`; no son llamados por el experimento final.

### Documentación interna e histórica

- `docs/maintenance/`: auditorías, walkthroughs y planes internos. Se verificó que no son leídos
  por código y contienen rutas personales o planes superados.
- `docs/parte_1/`, `docs/parte_2/` y `docs/parte_3/`: bitácoras, cierres anteriores y planes de
  segmentación/context stitching. No intervienen en ejecución; los resultados vigentes se
  conservarán como artefactos generados bajo `results/`.

### Ambientes duplicados y dependencias no usadas

- `environment-linux-gpu.yml`: variante para segmentación GPU; el experimento final configura UNI
  explícitamente en CPU. Se comprobó en `v41_config` y al crear el extractor compartido.
- `requirements.txt`: segunda lista incompleta y divergente respecto de Conda. Se conservará una
  sola fuente reproducible, `environment.yml`.

### Resultados temporales y archivos generados

- Árboles locales ignorados bajo `outputs/` salvo los agregados finales: contienen patches,
  previews por WSI, caches `.npz`, logs y corridas intermedias. Los scripts finales pueden
  regenerarlos; solo los agregados pequeños usados para verificar las cifras se versionarán.
- `.DS_Store`, `__pycache__/` y archivos `.pyc`: metadata de macOS y caches de Python; no existen
  referencias de código y ya están cubiertos por `.gitignore`.
- Los `.gitkeep` de carpetas de salidas vacías (`masks`, `overlays`, `segmentation`, `model_checks`,
  etc.): pertenecen a pipelines eliminados y no son requeridos por los scripts, que crean sus
  carpetas con `mkdir`.

## Comprobaciones previas a la eliminación

- Se parsearon todos los archivos Python con `ast` y se listaron todos sus imports.
- Se buscaron referencias a scripts numerados, módulos de selección históricos, columnas CSV,
  claves JSON, rutas absolutas y nombres de carpetas.
- Se revisaron los commits que introdujeron v4.1, el ejecutor de nueve WSI y el análisis
  morfológico.
- Se verificó que `inf402_n9` contiene nueve casos, 16 patches por método y agregados técnicos,
  espaciales, morfológicos y temporales.
- No se ha eliminado ni movido ningún archivo antes de registrar este inventario.
