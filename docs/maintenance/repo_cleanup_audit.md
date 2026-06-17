# Auditoria de limpieza del repositorio Lumina/Histora

Fecha de auditoria: 2026-06-16  
Repositorio: `/Users/davidkripper/INF402-Lumina-segmentation`

## Resumen ejecutivo

El repositorio tiene una separacion razonable entre pipeline actual, soporte
tecnico, flujo legacy de imagenes pequenas, placeholders y outputs locales. No
se detectaron archivos no trackeados no ignorados antes de crear este reporte.
El mayor riesgo de desorden esta en `outputs/`, que ocupa 505M y contiene
resultados locales ignorados por Git. No se recomienda limpiar esos outputs sin
confirmacion, porque contienen evidencia experimental reciente.

El pipeline actual debe mantenerse centrado en seleccion inteligente de patches
y segmentacion semantica tecnica posterior. Este proyecto no diagnostica, no
calcula RCB, no reemplaza al patologo y no constituye validacion clinica.

Hallazgos principales:

- `scripts/05_evaluate_bcss.py` y `src/inference/run_tiatoolbox_baseline.py`
  son placeholders explicitos.
- `scripts/03_extract_patches.py`, `src/patching/`, `src/config/paths.py` y
  `notebooks/00_exploration.ipynb` corresponden al flujo legacy de imagenes
  pequenas. Se recomienda archivar, no borrar directamente.
- `src/visualization/patch_preview.py` no debe moverse junto con legacy por
  ahora: tambien lo usan `src/selection/previews.py` y
  `src/preprocessing/wsi_patch_extraction.py`.
- `scripts/05_extract_wsi_patches.py` sigue siendo soporte util para extraccion
  y debug WSI acotado.
- `src/evaluation/metrics.py` debe conservarse como soporte futuro para
  evaluacion con ground truth/BCSS.

## Estado de Git

Estado observado al inicio de la auditoria:

```text
branch: main
git status --short: sin salida
tracked files: 64
find . -maxdepth 3: 156 entradas
git clean -nd: sin salida
```

Interpretacion:

- No habia cambios pendientes ni archivos no trackeados no ignorados al inicio
  de la auditoria.
- La creacion de este reporte agrego `docs/maintenance/repo_cleanup_audit.md`
  como archivo no trackeado hasta que el usuario decida stagear/commitear.
- No se ejecuto `git clean` sin `-n`.
- No se hizo `git stage`, commit ni push.

## Tamano por carpeta

Salida resumida de `du -sh ./*`:

| path | tamano |
|---|---:|
| `LICENSE` | 12K |
| `README.md` | 24K |
| `data/` | 16K |
| `docs/` | 40K |
| `environment-linux-gpu.yml` | 4.0K |
| `environment.yml` | 4.0K |
| `notebooks/` | 4.0K |
| `outputs/` | 505M |
| `requirements.txt` | 4.0K |
| `scripts/` | 140K |
| `src/` | 848K |

Salida resumida de carpetas relevantes:

| path | tamano | observacion |
|---|---:|---|
| `outputs/segmentation/` | 297M | outputs locales de inferencia y comparacion tecnica |
| `outputs/patch_selection/` | 101M | corridas locales de seleccion y comparacion |
| `outputs/final_patch_selection/` | 66M | seleccion final local |
| `outputs/wsi_patches/` | 18M | extraccion WSI local |
| `outputs/patches/` | 12M | pruebas legacy de imagen pequena |
| `outputs/inference_smoke/` | 11M | smoke tests locales |
| `outputs/model_checks/` | 4.0K | estado local de carga de modelo |
| `docs/parte_1/` | 4.0K | placeholder estructural |
| `docs/parte_2/` | 36K | documentacion actual de codigo/resultados |
| `scripts/__pycache__/` | 68K | cache Python ignorado |
| `src/` | 848K | codigo fuente versionado |

## Ignorados y no trackeados

`git clean -nd` no reporto archivos no trackeados no ignorados.

Resumen de `git clean -ndX` sin listar cientos de archivos individuales:

| grupo | tipo | tamano observado | accion sugerida |
|---|---|---:|---|
| `.DS_Store`, `outputs/.DS_Store`, `outputs/*/.DS_Store` | cruft OS local | pequeno | eliminar solo con confirmacion o limpieza local segura |
| `scripts/__pycache__/`, `src/**/__pycache__/` | cache Python | pequeno | eliminar en limpieza local segura |
| `outputs/final_patch_selection/` | output local ignorado | 66M | conservar hasta cerrar entregables |
| `outputs/inference_smoke/` | output local ignorado | 11M | limpiar solo si ya no se necesita reproducir smoke tests |
| `outputs/model_checks/` | output local ignorado | 4.0K | conservar como evidencia local o regenerar cuando sea necesario |
| `outputs/patch_selection/` | output local ignorado | 101M | conservar hasta documentar resultados finales |
| `outputs/patches/` contenidos generados | output local ignorado | 12M | limpiar solo con confirmacion |
| `outputs/segmentation/` | output local ignorado | 297M | conservar hasta cerrar comparacion tecnica |
| `outputs/wsi_patches/` | output local ignorado | 18M | conservar si se necesita trazabilidad del smoke WSI |

## Evidencia de uso

Busquedas ejecutadas:

- `src.patching`, `iter_patches_with_metadata`, `compute_tissue_ratio`:
  aparece en `scripts/03_extract_patches.py` y `src/patching/`.
- `src.config.paths`, `ensure_directories`, `PATCHES_DIR`:
  aparece en `scripts/03_extract_patches.py`, `notebooks/00_exploration.ipynb`
  y `src/config/paths.py`.
- `src.io`, `read_image`:
  solo aparece dentro de `src/io/slide_reader.py`.
- `pixel_accuracy`, `mean_iou`:
  solo aparece dentro de `src/evaluation/metrics.py`.
- `run_tiatoolbox_baseline`, `describe_baseline`:
  solo aparece dentro de `src/inference/run_tiatoolbox_baseline.py`.
- `05_extract_wsi_patches`:
  aparece en `README.md` y `docs/parte_2/plan_codigo.md`.
- `patch_preview`, `PatchBox`, `save_patch_selection_preview`:
  aparece en `scripts/03_extract_patches.py`, `src/selection/previews.py`,
  `src/preprocessing/wsi_patch_extraction.py` y
  `src/visualization/patch_preview.py`.

Conclusiones de uso:

- No eliminar nada usado por scripts activos.
- No mover `src/visualization/patch_preview.py` sin antes refactorizar
  `src/selection/previews.py` y `src/preprocessing/wsi_patch_extraction.py`.
- No eliminar `src/preprocessing/wsi_patch_extraction.py`; alimenta el flujo WSI
  y tambien es dependencia indirecta de seleccion.
- No eliminar `src/evaluation/metrics.py`; aunque todavia no esta integrado,
  corresponde al soporte futuro de evaluacion con ground truth/BCSS.

## Tabla de clasificacion

| path | tipo | estado_recomendado | razon | accion_sugerida | riesgo |
|---|---|---|---|---|---|
| `README.md` | documentacion | KEEP_SUPPORT | describe flujo actual, instalacion y etapas | mantener; actualizar solo si se reorganiza el repo | bajo |
| `.gitignore` | configuracion | KEEP_SUPPORT | protege datos, WSI, checkpoints y outputs | mantener; revisar si se quieren `.gitkeep` para nuevas carpetas de outputs | bajo |
| `environment.yml` | ambiente | KEEP_SUPPORT | ambiente Conda principal | mantener | bajo |
| `environment-linux-gpu.yml` | ambiente | KEEP_SUPPORT | ambiente para servidor/GPU | mantener | bajo |
| `requirements.txt` | ambiente | KEEP_SUPPORT | respaldo pip | mantener | bajo |
| `data/` | estructura datos | KEEP_SUPPORT | estructura versionada solo con `.gitkeep` | mantener; no agregar datos reales | alto si se versionan datos sensibles |
| `outputs/` | outputs locales | LOCAL_OUTPUT_IGNORE | contiene resultados locales ignorados, 505M | listar y conservar; limpiar solo con confirmacion | alto si se borra evidencia experimental |
| `outputs/segmentation/` | outputs locales | LOCAL_OUTPUT_IGNORE | segmentaciones y comparacion tecnica, 297M | conservar hasta cerrar reporte/resultados | alto |
| `outputs/patch_selection/` | outputs locales | LOCAL_OUTPUT_IGNORE | corridas de seleccion y comparacion, 101M | conservar hasta cerrar resultados finales | alto |
| `outputs/final_patch_selection/` | outputs locales | LOCAL_OUTPUT_IGNORE | seleccion final local, 66M | conservar hasta congelar entregables | alto |
| `outputs/wsi_patches/` | outputs locales | LOCAL_OUTPUT_IGNORE | extracciones WSI locales, 18M | limpiar solo con confirmacion | medio |
| `outputs/inference_smoke/` | outputs locales | LOCAL_OUTPUT_IGNORE | smoke tests locales, 11M | limpiar solo con confirmacion | medio |
| `outputs/model_checks/` | outputs locales | LOCAL_OUTPUT_IGNORE | estado local de carga de modelo | regenerable, pero conservar si sirve de evidencia | bajo |
| `.DS_Store` | cruft local | DELETE_CANDIDATE | archivo OS ignorado | eliminar en limpieza local segura posterior | bajo |
| `scripts/__pycache__/`, `src/**/__pycache__/` | cache local | DELETE_CANDIDATE | cache Python ignorado | eliminar en limpieza local segura posterior | bajo |
| `scripts/01_check_environment.py` | script pipeline | KEEP_CORE | verificacion de ambiente | mantener | bajo |
| `scripts/02_test_tiatoolbox_model.py` | script pipeline | KEEP_CORE | prueba carga modelo BCSS | mantener | bajo |
| `scripts/04_run_inference.py` | script pipeline | KEEP_CORE | smoke test inferencia sobre imagen/patch | mantener | bajo |
| `scripts/06_select_wsi_patches.py` | script pipeline | KEEP_CORE | entrada formal para seleccion baseline/smart | mantener | alto si se rompe |
| `scripts/07_compare_patch_selectors.py` | script pipeline | KEEP_CORE | comparacion formal de selectores | mantener | alto si se rompe |
| `scripts/08_segment_selected_patches.py` | script pipeline | KEEP_CORE | segmentacion tecnica sobre patches seleccionados | mantener | alto si se rompe |
| `scripts/09_compare_segmentation_on_selected_patches.py` | script pipeline | KEEP_CORE | comparacion tecnica de segmentacion | mantener | alto si se rompe |
| `src/selection/` | modulo pipeline | KEEP_CORE | seleccion baseline, smart v1/v2, manifests, previews, comparacion | mantener | alto |
| `src/inference/tiatoolbox_inference.py` | modulo pipeline | KEEP_CORE | wrapper reusable de inferencia TIAToolbox | mantener | alto |
| `src/inference/selected_patch_segmentation.py` | modulo pipeline | KEEP_CORE | segmenta batches de patches seleccionados | mantener | alto |
| `src/inference/segmentation_comparison.py` | modulo pipeline | KEEP_CORE | compara outputs tecnicos de segmentacion | mantener | alto |
| `src/models/tiatoolbox_bcss.py` | modulo pipeline | KEEP_CORE | carga/configuracion modelo BCSS | mantener | alto |
| `src/visualization/segmentation_overlay.py` | modulo pipeline | KEEP_CORE | overlays/leyendas para segmentacion | mantener | alto |
| `scripts/05_extract_wsi_patches.py` | soporte WSI | KEEP_SUPPORT | extractor/debug WSI acotado; documentado en README/docs | mantener | medio |
| `src/preprocessing/wsi_patch_extraction.py` | soporte WSI | KEEP_SUPPORT | usado por script 05 y modulos de seleccion | mantener | alto |
| `src/preprocessing/tissue_detection.py` | soporte filtrado | KEEP_SUPPORT | usado por patch filtering legacy y soporte de tejido | mantener | medio |
| `src/visualization/patch_preview.py` | soporte visual | KEEP_SUPPORT | usado por selection/previews y wsi_patch_extraction, no solo legacy | mantener; revisar solo en refactor posterior | medio |
| `src/evaluation/metrics.py` | soporte futuro | KEEP_SUPPORT | contiene pixel accuracy y mean IoU para evaluacion futura BCSS | mantener | medio |
| `docs/parte_2/` | documentacion | KEEP_SUPPORT | planes, resultados y auditoria de patch selection | mantener | bajo |
| `scripts/03_extract_patches.py` | legacy small-image | ARCHIVE_CANDIDATE | flujo inicial para imagenes pequenas; aun documentado en README | mover a `archive/legacy_small_image_pipeline/` solo con confirmacion | medio |
| `src/patching/` | legacy small-image | ARCHIVE_CANDIDATE | usado por script 03; no es core WSI actual | archivar con script 03 solo con confirmacion | medio |
| `src/config/paths.py` | legacy/config inicial | ARCHIVE_CANDIDATE | usado por script 03 y notebook; no por pipeline WSI actual | archivar con flujo legacy o mantener si se quiere config central | medio |
| `notebooks/00_exploration.ipynb` | exploracion inicial | ARCHIVE_CANDIDATE | reservado para pruebas pequenas; solo llama `ensure_directories` | archivar o dejar como ejemplo minimo, con confirmacion | bajo |
| `scripts/05_evaluate_bcss.py` | placeholder | PLACEHOLDER_REMOVE_CANDIDATE | placeholder explicito de evaluacion BCSS futura | reemplazar por implementacion real o eliminar con confirmacion | bajo |
| `src/inference/run_tiatoolbox_baseline.py` | placeholder | PLACEHOLDER_REMOVE_CANDIDATE | placeholder historico; funcionalidad real vive en otros wrappers | eliminar o archivar con confirmacion | bajo |
| `docs/parte_1/.gitkeep` | estructura docs | NEEDS_CONFIRMATION | carpeta vacia versionada; puede apuntar a docs externos OneDrive | confirmar si se mantiene como placeholder | bajo |
| `src/io/slide_reader.py` | helper no usado | NEEDS_CONFIRMATION | `read_image` no tiene referencias externas | confirmar si se elimina, archiva o se integra | bajo |
| `src/visualization/overlays.py` | helper no usado | NEEDS_CONFIRMATION | overlay simple no usado por pipeline actual | confirmar si se elimina, archiva o se integra | bajo |

## Limpieza propuesta por etapas

### Etapa A: limpieza local segura

No ejecutar durante esta auditoria. En una tarea posterior, con confirmacion
explicita, eliminar solo cruft ignorado y regenerable:

- `.DS_Store`.
- `scripts/__pycache__/`.
- `src/**/__pycache__/`.

No tocar `outputs/` en esta etapa salvo que el usuario confirme exactamente que
carpetas quiere limpiar.

### Etapa B: archivar flujo legacy de imagenes pequenas

No borrar directamente. Si el equipo decide que el flujo small-image ya no debe
estar en la raiz operativa, mover con confirmacion a:

```text
archive/legacy_small_image_pipeline/
```

Candidatos:

- `scripts/03_extract_patches.py`.
- `src/patching/`.
- `src/config/paths.py`.
- `notebooks/00_exploration.ipynb`.

Antes de mover, actualizar referencias en README/notebook o dejar una nota de
archivo para que no queden comandos rotos.

### Etapa C: alinear README y `.gitignore`

El README todavia documenta flujos historicos utiles, pero podria separarlos en
una seccion "legacy/debug". `.gitignore` esta razonablemente alineado porque
ignora datos, WSI, checkpoints y outputs. Solo revisar si se quieren `.gitkeep`
para carpetas nuevas como:

- `outputs/patch_selection/`.
- `outputs/final_patch_selection/`.
- `outputs/segmentation/`.
- `outputs/wsi_patches/`.
- `outputs/model_checks/`.

No agregar outputs reales al repositorio.

### Etapa D: limpiar outputs locales con confirmacion

No hacer en esta auditoria. Si se limpia, hacerlo por carpeta y despues de
respaldar o documentar lo necesario:

- `outputs/segmentation/` si ya se exportaron resultados finales.
- `outputs/patch_selection/` si ya no se necesitan corridas intermedias.
- `outputs/final_patch_selection/` solo despues de cerrar entregables.
- `outputs/inference_smoke/` y `outputs/wsi_patches/` si son regenerables.

## Comandos ejecutados

```bash
git status --short
git branch --show-current
git ls-files
find . -maxdepth 3 -not -path "./.git/*" | sort
du -sh ./* 2>/dev/null
du -sh data/* outputs/* docs/* notebooks/* scripts/* src/* 2>/dev/null
git clean -nd
git clean -ndX
grep -R "src.patching\|from src.patching\|iter_patches_with_metadata\|compute_tissue_ratio" -n . --exclude-dir=.git --exclude-dir=outputs --exclude-dir=data
grep -R "src.config.paths\|ensure_directories\|PATCHES_DIR" -n . --exclude-dir=.git --exclude-dir=outputs --exclude-dir=data
grep -R "src.io\|read_image" -n . --exclude-dir=.git --exclude-dir=outputs --exclude-dir=data
grep -R "pixel_accuracy\|mean_iou" -n . --exclude-dir=.git --exclude-dir=outputs --exclude-dir=data
grep -R "run_tiatoolbox_baseline\|describe_baseline" -n . --exclude-dir=.git --exclude-dir=outputs --exclude-dir=data
grep -R "05_evaluate_bcss\|05_extract_wsi_patches\|03_extract_patches" -n README.md docs scripts src --exclude-dir=.git
grep -R "patch_preview\|PatchBox\|save_patch_selection_preview" -n README.md docs scripts src notebooks --exclude-dir=.git --exclude-dir=outputs --exclude-dir=data --exclude-dir=__pycache__
```

Comandos deliberadamente no ejecutados:

- `git clean -fd`.
- `git clean -fdX`.
- `git add`.
- `git commit`.
- `git push`.
- Cualquier `rm`, `mv` o limpieza destructiva.

## Confirmacion de alcance

Esta auditoria solo creo este reporte. No se borro ningun archivo, no se movio
ninguna carpeta, no se modifico codigo fuente, no se limpiaron outputs locales,
no se stageo, no se commiteo y no se pusheo nada.
