# Plan iHealth: selector orientado a tumor residual y segmentación robusta

Fecha de referencia: 2026-06-16  
Repositorio: `/Users/davidkripper/INF402-Lumina-segmentation`

Este documento propone una evolución arquitectónica del pipeline Lumina/Histora
si se asume que la ejecución se realizará en servidores de iHealth. No describe
una implementación ya existente ni constituye validación clínica. El objetivo
es orientar decisiones técnicas futuras para seleccionar patches más útiles para
una segmentación posterior relacionada con análisis técnico de tumor residual.

Alcance prudente:

- no diagnostica;
- no calcula RCB;
- no reemplaza al patólogo;
- no valida clínicamente resultados;
- no interpreta predicciones como ground truth clínico;
- no promete detectar tumor con certeza clínica.

## 1. Cambio de supuesto computacional

El pipeline actual fue diseñado con una restricción fuerte: debía correr de
forma reproducible y razonable en un entorno local CPU-friendly, incluyendo un
MacBook con memoria limitada. Esa decisión fue correcta para construir una base
estable: selección desde WSI, comparación de selectores, segmentación técnica
sobre patches seleccionados y comparación de outputs de segmentación.

Si el supuesto cambia y la ejecución se realizará siempre en servidores de
iHealth, las prioridades también cambian. El cuello de botella deja de ser
solamente evitar RAM/CPU local y pasa a ser:

- calidad de selección;
- robustez operacional;
- trazabilidad de decisiones;
- compatibilidad con WSI reales;
- valor para una demo técnica de feria;
- facilidad de integración con revisión experta.

No es necesario optimizar para procesar muchas WSI simultáneamente desde el
inicio. El caso de uso principal puede ser una WSI a la vez, con más presupuesto
para leer candidatos, calcular features, ejecutar modelos auxiliares y guardar
evidencia intermedia. Bajo este supuesto, conviene priorizar la utilidad de los
patches seleccionados para segmentación posterior, no solo la velocidad local.

Esto no significa volver el proyecto ilimitado. Aun con servidores, siguen
existiendo restricciones reales: tiempo de integración, disponibilidad de GPU,
I/O sobre WSI grandes, estabilidad de dependencias, almacenamiento, revisión de
outputs y necesidad de mantener comparaciones defendibles.

## 2. Estado actual del selector

El repo contiene tres selectores de patches integrados en
`scripts/06_select_wsi_patches.py`.

### `baseline_tiatoolbox`

`baseline_tiatoolbox` vive en `src/selection/tiatoolbox_baseline.py`. Genera
una grilla de candidatos, estima una máscara técnica de tejido a partir de un
thumbnail, filtra por `min_tissue_ratio`, baraja con `seed` y lee patches reales
hasta cumplir `max_patches`.

Qué hace bien:

- es reproducible;
- define un baseline simple y auditable;
- separa `candidate_metadata.csv` como pool común thumbnail-filtered;
- guarda `selected_metadata.csv` solo con patches seleccionados;
- permite comparar otros selectores bajo el mismo presupuesto.

Limitaciones:

- no usa deep learning;
- no usa el modelo de segmentación para seleccionar;
- no estima probabilidad de clases relevantes;
- no entiende heterogeneidad morfológica más allá de máscara de tejido;
- puede seleccionar patches redundantes si muchas regiones cumplen el filtro.

### `smart_tissue_nuclei_v1`

`smart_tissue_nuclei_v1` vive en `src/selection/smart_tissue_nuclei.py`. Usa el
mismo pool común de candidatos, lee una muestra controlada por
`max_candidates_to_score` y calcula features simples: proporción de tejido,
proxy nuclear RGB, entropía visual, blur y penalización de artefactos. Luego
aplica un score y diversidad espacial greedy.

Qué hace bien:

- prioriza patches más informativos que una grilla uniforme;
- mantiene bajo el costo local;
- mejora trazabilidad por candidato;
- sirve como ablation entre baseline y v2.

Limitaciones:

- no usa deep learning;
- no usa segmentación preliminar;
- no identifica clases histológicas específicas;
- usa proxies visuales, no detección real de tumor;
- puede quedarse corto si el objetivo es utilidad para análisis técnico de
  tumor residual.

### `smart_tissue_nuclei_v2_light`

`smart_tissue_nuclei_v2_light` extiende v1 con HED color deconvolution como
proxy nuclear, cuotas espaciales suaves y diversidad por features. Sigue siendo
CPU-friendly y no depende de modelos deep learning.

Qué hace bien:

- mejora la señal nuclear con HED;
- reduce concentración espacial mediante cuotas;
- agrega diversidad visual simple;
- mantiene compatibilidad con la comparación formal existente.

Limitaciones:

- HED y features visuales siguen siendo proxies técnicos;
- no detecta tumor;
- no infiere clases relevantes antes de seleccionar;
- no usa embeddings histopatológicos profundos;
- no optimiza explícitamente para utilidad de segmentación orientada a análisis
  técnico de tumor residual.

### Relación con TIAToolbox

El modelo `fcn_resnet50_unet-bcss` se usa después de la selección, mediante
`scripts/04_run_inference.py` y `scripts/08_segment_selected_patches.py`. No se
usa actualmente para decidir qué patches seleccionar. Por eso, los selectores
actuales seleccionan patches por heurísticas técnicas y proxies visuales, no por
clases predichas por el modelo de segmentación.

## 3. Nuevo objetivo del selector

La evolución propuesta es crear un selector conceptual llamado:

```text
tumor_residual_candidate_selector_v1
```

Nombre alternativo más prudente si se prefiere evitar sobrelectura clínica:

```text
residual_analysis_candidate_selector_v1
```

Objetivo:

Seleccionar patches con mayor utilidad esperada para una segmentación posterior
relacionada con análisis técnico de tumor residual, manteniendo trazabilidad y
sin afirmar diagnóstico, RCB ni validación clínica.

El selector debería priorizar:

- patches con tejido informativo;
- patches con probabilidad técnica de contener clases relevantes para
  segmentación;
- regiones frontera o heterogéneas cuando aporten información;
- diversidad espacial;
- diversidad morfológica;
- baja redundancia;
- bajo blur;
- baja presencia de artefactos;
- exclusión de fondo y tejido poco informativo.

La salida debe seguir siendo una selección bajo presupuesto. El selector no
debe intentar procesar toda la WSI como si cada patch tuviera el mismo valor.
Debe producir un subconjunto defendible, auditable y útil para alimentar la
segmentación posterior.

## 4. Arquitectura propuesta del nuevo selector

La arquitectura recomendada es por etapas. Esto permite mantener parte del
pipeline actual y reemplazar solo las piezas que agregan valor.

### Etapa 1: screening técnico de WSI

Responsabilidad: construir un pool amplio de candidatos técnicamente válidos.

Componentes:

- generación densa de candidatos sobre level 0 o sobre una magnificación
  definida;
- máscara de tejido basada en thumbnail;
- filtros de calidad para fondo, regiones vacías, blur y artefactos;
- cálculo de features técnicas actuales;
- metadata completa por candidato.

Entradas:

- WSI;
- `patch_size`;
- `stride`;
- magnificación/resolución objetivo;
- `min_tissue_ratio`;
- parámetros de calidad.

Salidas:

- pool de candidatos filtrados;
- features técnicas;
- trazabilidad espacial;
- razón de descarte cuando corresponda.

Ventajas:

- conserva la base actual;
- separa calidad técnica de relevancia histológica;
- permite comparar contra baseline y smart v2.

Desventajas:

- por sí sola no sabe si un patch será útil para análisis técnico de tumor
  residual;
- si los filtros son agresivos, pueden descartar regiones raras o difíciles.

### Etapa 2: ranking model-assisted

Responsabilidad: estimar utilidad esperada para segmentación posterior.

Alternativa A: embeddings de patches con modelo histopatológico preentrenado.

La idea es calcular vectores de representación para candidatos y usar esos
embeddings para diversidad morfológica, clustering y ranking.

Ventajas:

- captura patrones visuales más ricos que HED/entropía/blur;
- permite clustering representativo;
- ayuda a reducir redundancia morfológica;
- no requiere que el mismo modelo de segmentación defina todo el ranking.

Desventajas:

- requiere elegir, instalar y validar un modelo;
- puede introducir dependencias pesadas;
- necesita batch inference y cache de embeddings;
- la interpretabilidad puede ser menor que con features simples.

Alternativa B: clasificador o ranker de patch informativo.

La idea es entrenar o calibrar un modelo que prediga si un patch es útil para la
segmentación posterior.

Ventajas:

- alinea directamente selección con utilidad esperada;
- puede incorporar feedback de patólogo;
- puede mejorar con datos internos.

Desventajas:

- necesita labels o feedback confiable;
- puede sobreajustarse a pocas WSI;
- requiere protocolo de evaluación cuidadoso.

Alternativa C: inferencia preliminar de segmentación en baja resolución o sobre
candidatos.

La idea es usar una pasada rápida del modelo de segmentación para obtener
pseudo-labels, distribución de clases predichas, entropía de predicción o
presencia de clases potencialmente informativas.

Ventajas:

- conecta selección con la tarea posterior;
- permite priorizar patches con clases predichas relevantes;
- puede ser natural en servidores con GPU;
- reutiliza infraestructura de inferencia.

Desventajas:

- riesgo de circularidad si se evalúa con el mismo modelo;
- errores del modelo pueden sesgar el selector;
- puede perder regiones importantes si el modelo falla temprano;
- exige separar claramente pseudo-labels de ground truth.

Alternativa D: combinación de features actuales con scores de modelo.

La idea es mantener `tissue_ratio`, HED, blur, artefactos y diversidad, pero
agregar uno o más scores model-assisted.

Ventajas:

- minimiza riesgo de integración;
- mantiene interpretabilidad;
- permite ablation;
- permite apagar el componente model-assisted si falla.

Desventajas:

- requiere definir pesos;
- puede ser difícil justificar pesos sin ground truth;
- los componentes pueden estar correlacionados.

Recomendación: comenzar por la alternativa D, con una implementación modular
que permita activar embeddings o segmentación preliminar como backends
opcionales.

### Etapa 3: selección multiobjetivo

Responsabilidad: transformar scores y constraints en una lista final de patches.

El selector debería combinar:

- score de relevancia;
- score de calidad técnica;
- diversidad espacial;
- diversidad visual o por embedding;
- cobertura de regiones potencialmente relevantes;
- penalización por redundancia;
- presupuesto fijo de patches por WSI.

Una fórmula posible:

```text
score_final =
  w_relevance * model_relevance_score
  + w_quality * technical_quality_score
  + w_nuclear * nuclear_or_cellularity_score
  + w_entropy * heterogeneity_score
  + w_embedding * embedding_diversity_bonus
  - w_artifact * artifact_penalty
  - w_redundancy * redundancy_penalty
```

La fórmula exacta debe quedar en `method_config.json` y en el summary de la
corrida. Si se usa una etapa greedy, cada decisión debería ser reproducible por
seed y quedar registrada por candidato.

## 5. ¿Debe el selector usar el modelo de segmentación?

Hoy el selector no usa el modelo de segmentación. El modelo
`fcn_resnet50_unet-bcss` se aplica después, sobre los patches ya seleccionados.
Esto mantiene una separación limpia entre selección y segmentación.

En un entorno iHealth, sí puede ser razonable usar una pasada preliminar del
modelo de segmentación como parte del ranking. Por ejemplo:

- ejecutar inferencia preliminar sobre candidatos;
- calcular distribución de clases predichas;
- priorizar patches con clases predichas consideradas informativas;
- medir entropía o heterogeneidad de la predicción;
- detectar patches donde el modelo genera regiones mixtas o fronteras.

Pero debe documentarse como un enfoque coarse-to-fine:

```text
screening tecnico -> prediccion preliminar -> seleccion -> segmentacion final
```

La predicción preliminar no es ground truth. Es una señal técnica para priorizar
recursos.

Riesgo principal: circularidad. Si el selector usa el mismo modelo de
segmentación que luego se evalúa, la comparación puede favorecer al selector
porque está optimizado para las mismas predicciones. Esto no invalida el método,
pero cambia la interpretación: se evalúa consistencia técnica con el modelo, no
descubrimiento independiente de verdad histológica.

Mitigaciones:

- registrar explícitamente si el selector usó el modelo de segmentación;
- comparar también métricas independientes de selección, como diversidad,
  cobertura espacial y calidad técnica;
- usar ground truth externo cuando exista;
- separar ablations: sin modelo, con embeddings, con segmentación preliminar;
- evitar presentar pseudo-labels como anotaciones clínicas;
- considerar un modelo auxiliar/ranker separado para selección.

Recomendación: permitir que el selector use el modelo de segmentación como
backend opcional de scoring, no como única base del selector.

## 6. Diseño incremental recomendado

### v3_server_quality

Objetivo: mejorar el selector actual sin introducir modelos nuevos.

Características:

- scorear muchos más candidatos;
- leer más patches reales;
- usar `feature_size` mayor cuando el servidor lo permita;
- mejorar filtros de blur, artefactos y fondo;
- mantener HED como proxy nuclear;
- usar cuotas espaciales y diversidad por features;
- guardar más metadata intermedia;
- permitir cache/reanudación.

Ventajas:

- bajo riesgo de integración;
- continuidad directa con `smart_tissue_nuclei_v2_light`;
- fácil de explicar en informe;
- sirve como baseline fuerte antes de usar modelos.

Limitación:

- sigue sin saber directamente qué clases predecirá el modelo de segmentación.

### v4_model_assisted

Objetivo: agregar scoring preliminar basado en modelo.

Opciones:

- embeddings histopatológicos para clustering/diversidad;
- inferencia preliminar de TIAToolbox;
- ranker separado de utilidad esperada;
- combinación de features actuales con score de modelo.

Outputs esperados:

- `model_relevance_score`;
- `predicted_class_distribution` si se usa segmentación preliminar;
- `embedding_cluster_id` si se usa clustering;
- `embedding_diversity_score`;
- `selector_backend`;
- `selector_backend_model_name`;
- `selector_backend_warning`.

Ventajas:

- más alineado con utilidad para segmentación posterior;
- permite seleccionar patches potencialmente más relevantes;
- aprovecha servidores iHealth.

Limitación:

- requiere protocolo claro para evitar sobreinterpretación y circularidad.

### v5_pathologist_feedback

Objetivo: incorporar feedback experto como señal de calibración.

Ideas:

- guardar decisiones de revisión: patch útil, no útil, dudoso;
- registrar motivo: artefacto, pobre celularidad, región informativa, frontera,
  clase de interés predicha;
- permitir exportar un set de patches para revisión;
- usar feedback para ajustar pesos o entrenar un ranker;
- separar feedback de patólogo de predicciones automáticas.

Ventajas:

- mejora alineación con uso real;
- permite validar utilidad práctica de los patches;
- prepara una demo más convincente.

Limitación:

- requiere disponibilidad de patólogos y protocolo de anotación simple.

## 7. Cambios esperados en código

No se propone implementar nada en esta etapa. Si se aprueba el plan, una
evolución ordenada podría agregar o modificar los siguientes archivos.

### `src/selection/tumor_residual_selector.py`

Responsabilidad: implementar `tumor_residual_candidate_selector_v1` o un nombre
más prudente como `residual_analysis_candidate_selector_v1`.

Debería contener:

- config del selector;
- validación de parámetros;
- generación/lectura del pool de candidatos;
- orquestación de scoring técnico y model-assisted;
- selección multiobjetivo;
- escritura de metadata, summary y config.

### `src/selection/model_assisted_scoring.py`

Responsabilidad: calcular scores derivados de modelos.

Podría incluir:

- score por distribución de clases predichas;
- score por entropía de predicción;
- score por presencia de clases informativas;
- helpers para correr inferencia preliminar controlada;
- serialización de pseudo-labels o estadísticas por candidato.

### `src/selection/embedding_scoring.py`

Responsabilidad: calcular embeddings y métricas de diversidad morfológica.

Podría incluir:

- carga de modelo de embeddings;
- batch inference;
- cache de vectores;
- clustering;
- score de representatividad;
- score de diversidad.

### `src/selection/server_config.py`

Responsabilidad: agrupar parámetros de ejecución en servidor.

Podría incluir:

- batch size;
- workers;
- device;
- cache dirs;
- límites de candidatos;
- flags de resume/checkpointing;
- modo debug/minimal/full.

### `scripts/10_select_tumor_residual_patches.py`

Responsabilidad: CLI del nuevo selector.

Debería seguir el estilo de `scripts/06_select_wsi_patches.py`:

- `ROOT_DIR`;
- imports desde `src`;
- `argparse`;
- `--overwrite` seguro;
- mensajes `[OK]`, `[FAIL]`, `[WARN]`;
- warning clínico prudente;
- salida compatible con `scripts/08_segment_selected_patches.py`.

### `docs/maintenance/ihealth_tumor_residual_selector_plan.md`

Responsabilidad: documento de arquitectura actual.

Debe mantenerse como referencia de diseño, no como evidencia de implementación.

### Modificaciones potenciales a archivos existentes

- `scripts/06_select_wsi_patches.py`: solo si se decide registrar el nuevo
  selector en el CLI existente.
- `src/selection/__init__.py`: exportar config y runner del nuevo selector.
- `src/selection/manifests.py`: agregar columnas opcionales para model-assisted
  scoring si se decide mantener un contrato CSV común.
- `scripts/07_compare_patch_selectors.py`: extender comparación si el nuevo
  selector agrega embeddings, pseudo-labels o scores de relevancia.
- `scripts/08_segment_selected_patches.py`: idealmente no debería cambiar si el
  nuevo selector mantiene `selected_metadata.csv` y `selected/`.

## 8. Configuración para iHealth

Parámetros que deberían quedar explícitos:

- `device`: `cuda`, `cpu` o `auto`;
- `batch_size`;
- `num_workers`;
- `tile_size` o `patch_size`;
- `stride`;
- magnificación o resolución objetivo;
- `max_candidates`;
- `max_candidates_to_score`;
- `max_patches`;
- modo de salida: `debug`, `minimal`, `full`;
- cache de features técnicas;
- cache de embeddings;
- cache de inferencia preliminar;
- `resume`;
- `checkpoint_interval`;
- carpeta de scratch temporal;
- nivel de logging;
- seed;
- política de overwrite seguro.

Aspectos por confirmar en iHealth:

- si existe scheduler tipo SLURM;
- GPUs disponibles;
- memoria CPU y GPU;
- número de workers recomendado;
- rutas de datasets;
- rutas de scratch;
- política de jobs largos;
- límites de almacenamiento;
- permisos para caches de modelos;
- compatibilidad con Conda, Mamba, Docker, Apptainer o módulos del sistema;
- versión de drivers/CUDA;
- disponibilidad de OpenSlide y librerías nativas.

Recomendación operativa: no asumir valores de servidor en el código. Usar
configuración externa y defaults conservadores. El summary de cada corrida debe
guardar los parámetros efectivos y el dispositivo resuelto.

## 9. Segmentación futura robusta

El foco de este documento es selección de patches, pero la selección solo tiene
sentido si alimenta una segmentación posterior confiable a nivel técnico.

Estado actual:

- el modelo usado es `fcn_resnet50_unet-bcss`;
- la inferencia se ejecuta con TIAToolbox;
- `scripts/04_run_inference.py` sirve para smoke test individual;
- `scripts/08_segment_selected_patches.py` segmenta todos los patches
  seleccionados de una corrida;
- `scripts/09_compare_segmentation_on_selected_patches.py` compara outputs de
  segmentación entre corridas;
- los outputs son máscaras, overlays, leyendas y summaries.

Limitaciones actuales:

- el flujo actual es más cercano a smoke/batch técnico sobre patches que a un
  sistema de inferencia WSI completo;
- no hay evaluación formal con ground truth integrada en el flujo principal;
- las métricas de comparación de segmentación son sobre clases predichas, no
  sobre verdad clínica;
- la máscara cruda puede tener resolución distinta al overlay visual;
- no hay fine-tuning integrado;
- no hay batch inference optimizada para GPU a gran escala;
- no hay tile overlap formal para reducir artefactos de borde.

Mejoras futuras:

- batch inference GPU;
- control explícito de magnificación y resolución;
- tile overlap y blending cuando corresponda;
- cache de outputs intermedios;
- manejo robusto de fallas por patch;
- evaluación con ground truth compatible;
- métricas Dice, IoU/mIoU y pixel accuracy cuando exista ground truth;
- comparación entre overlay visual y máscara cruda;
- posible fine-tuning o adaptación si el modelo preentrenado no es suficiente.

Fine-tuning debe quedar como etapa posterior. Requiere datos anotados,
separación train/validation/test, control de leakage por paciente/WSI y métricas
defendibles. No debe presentarse como requisito inmediato del selector.

## 10. Riesgos y decisiones pendientes

### Definir qué significa "patch útil"

Sin ground truth, "patch útil" no puede significar "patch clínicamente correcto".
Debe definirse operacionalmente, por ejemplo:

- patch con tejido suficiente;
- patch con baja presencia de artefactos;
- patch con diversidad visual;
- patch con clases predichas informativas por el modelo;
- patch que mejora cobertura espacial;
- patch que un patólogo marca como útil para revisión técnica.

### Evitar sesgo/circularidad

Si el selector usa el mismo modelo que luego se evalúa, el resultado puede medir
afinidad con ese modelo, no calidad independiente. Para mitigarlo:

- reportar claramente el backend usado;
- incluir ablations;
- comparar contra baseline bajo mismo presupuesto;
- incorporar métricas independientes;
- usar ground truth externo cuando exista;
- evitar lenguaje de validación clínica.

### Datos necesarios

El equipo debe aclarar:

- qué WSI estarán disponibles;
- si hay ground truth compatible;
- si hay metadata de magnificación/MPP confiable;
- si existen anotaciones de regiones relevantes;
- qué restricciones de uso o privacidad aplican;
- qué formato de salida necesita la demo.

### Feedback de patólogos

Sería valioso acordar una plantilla simple de feedback:

- patch útil/no útil/dudoso;
- razón principal;
- calidad técnica;
- presencia de región informativa;
- comentarios libres;
- decisión de incluir/excluir en demo.

Esto no reemplaza validación clínica. Sirve para calibrar utilidad práctica del
selector.

### Qué se puede validar antes de feria

Validaciones realistas:

- reproducibilidad de corridas;
- runtime y estabilidad en servidor;
- comparación baseline vs nuevo selector;
- distribución de clases predichas por modelo;
- diversidad espacial y morfológica;
- revisión visual de overlays;
- cobertura de regiones anotadas si existe ground truth compatible.

Lo que no se debe prometer:

- diagnóstico automático;
- cálculo completo de RCB;
- cuantificación clínica de cáncer residual;
- reemplazo del patólogo;
- generalización clínica;
- detección de tumor con certeza.

## 11. Roadmap recomendado

### Fase 1: documentar y diseñar selector server-quality

Objetivo: cerrar contrato técnico antes de escribir código.

Entregables:

- especificación de inputs/outputs;
- columnas nuevas de metadata;
- configuración para servidor;
- criterios de scoring;
- definición operacional de patch útil;
- protocolo de comparación contra baseline.

### Fase 2: implementar `v3_server_quality`

Objetivo: mejorar el selector sin modelos nuevos.

Cambios esperados:

- scorear más candidatos;
- aumentar resolución de features;
- mejorar filtros;
- incorporar cache/reanudación;
- mantener compatibilidad con outputs actuales;
- comparar contra `baseline_tiatoolbox` y `smart_tissue_nuclei_v2_light`.

### Fase 3: implementar ranking model-assisted

Objetivo: agregar señales de modelo de forma controlada.

Opciones:

- embeddings para diversidad morfológica;
- segmentación preliminar coarse-to-fine;
- score de relevancia por clases predichas;
- ranker calibrable con feedback.

Debe incluir ablations para separar el valor de cada componente.

### Fase 4: mejorar segmentación GPU/batch

Objetivo: que la segmentación posterior sea robusta y eficiente en servidor.

Cambios esperados:

- batch inference;
- manejo de GPU;
- tile overlap si corresponde;
- control de resolución;
- summaries más completos;
- evaluación con ground truth cuando exista.

### Fase 5: integrar feedback de patólogo y preparar demo feria

Objetivo: conectar el pipeline técnico con revisión experta y narrativa de
demo.

Entregables:

- set de patches seleccionados;
- overlays revisables;
- comparación baseline vs selector avanzado;
- registro de feedback;
- conclusiones prudentes;
- lista explícita de limitaciones.

## 12. Resumen ejecutivo final

La recomendación principal es no saltar directamente a un sistema complejo de
diagnóstico o cuantificación. El paso correcto es evolucionar desde
`smart_tissue_nuclei_v2_light` hacia un selector server-quality y luego hacia un
selector model-assisted modular.

Qué hacer ahora:

- mantener `baseline_tiatoolbox` como control;
- mantener `smart_tissue_nuclei_v2_light` como referencia CPU-friendly;
- diseñar `v3_server_quality` con más candidatos, mejores filtros y cache;
- definir el contrato de metadata para señales model-assisted;
- preparar una comparación clara bajo el mismo presupuesto de patches.

Qué dejar para después:

- embeddings profundos;
- segmentación preliminar como backend de scoring;
- ranker entrenado;
- feedback de patólogo;
- fine-tuning de segmentación.

Qué no hacer todavía:

- prometer detección clínica;
- presentar pseudo-labels como ground truth;
- calcular RCB;
- convertir el selector en una caja negra sin ablations;
- cambiar todo el pipeline antes de validar una versión incremental.

El objetivo defendible para iHealth es construir un selector que priorice
patches histológicamente informativos y técnicamente útiles para segmentación
posterior, con trazabilidad suficiente para revisión visual y comparación
experimental.
