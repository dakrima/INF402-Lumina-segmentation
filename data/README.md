# Datos externos

Para ejecutar las 9 WSIs, estas deben de estar almacenadas en el siguiente path:

```text
data/raw/wsi/
```

`scripts/ejecutar_experimento.py` espera exactamente nueve archivos y obtiene el identificador
de caso desde los tres primeros componentes separados por `-`. Los resultados usan
identificadores `TCGA-*`.