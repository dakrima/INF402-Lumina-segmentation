# Datos externos

Las nueve WSI no se versionan en este repositorio. Descargue los casos utilizados por el
experimento desde la fuente autorizada y ubique los archivos `.svs` directamente en:

```text
data/raw/wsi/
```

`scripts/ejecutar_experimento.py` espera exactamente nueve archivos y obtiene el identificador
de caso desde los tres primeros componentes separados por `-`. Los resultados preservados usan
identificadores `TCGA-*`; la fuente y denominación exactas del conjunto deben confirmarse contra
el manuscrito antes de publicar el README definitivo. No renombre las WSI para forzar otra
procedencia.

La descarga no se automatiza porque el acceso y la redistribución dependen de las condiciones
del proveedor del dataset.
