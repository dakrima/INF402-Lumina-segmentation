# Legacy small-image pipeline

Este directorio conserva el flujo historico usado para pruebas iniciales sobre
imagenes pequenas (`.png`, `.jpg`, `.tif`) antes de consolidar el pipeline
actual basado en WSI, seleccion inteligente de patches y segmentacion tecnica.

No forma parte de la ruta principal actual del proyecto. Se mantiene solo como
referencia historica y como material de consulta para experimentos simples.

Contenido archivado:

- `scripts/03_extract_patches.py`
- `src/patching/`
- `src/config/`
- `notebooks/00_exploration.ipynb`

Si alguien quiere ejecutar este flujo desde `archive/`, probablemente tendra que
ajustar imports y paths, porque el codigo fue movido fuera de su ubicacion
original. El pipeline operativo actual vive en:

- `scripts/06_select_wsi_patches.py`
- `scripts/07_compare_patch_selectors.py`
- `scripts/08_segment_selected_patches.py`
- `scripts/09_compare_segmentation_on_selected_patches.py`

Este flujo archivado no diagnostica, no calcula RCB, no reemplaza al patologo y
no constituye validacion clinica.
