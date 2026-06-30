# Datos externos

Las nueve WSI no se versionan en este repositorio. Descargue los casos utilizados por el
experimento desde la fuente autorizada y ubique los archivos `.svs` directamente en:

```text
data/raw/BACH/
```

`scripts/ejecutar_experimento.py` espera exactamente nueve archivos y obtiene el identificador
de caso desde los tres primeros componentes separados por `-` del nombre original TCGA. No
renombre las WSI.

La descarga no se automatiza porque el acceso y la redistribución dependen de las condiciones
del proveedor del dataset.
