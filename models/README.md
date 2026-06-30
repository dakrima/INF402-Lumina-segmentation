# Modelo UNI utilizado

El modelo UNI se tiene que descargar y autorizar por token a través del siguiente link: https://huggingface.co/MahmoodLab/UNI

Luego, se tiene que guardar en una carpeta con el siguiente path:

```text
models/UNI/pytorch_model.bin
```

También puede definir `UNI_MODEL_PATH` o usar `--uni-model-path`. Si la descarga requiere
autenticación de Hugging Face, entregue el token a la herramienta de descarga mediante
`HF_TOKEN`.
