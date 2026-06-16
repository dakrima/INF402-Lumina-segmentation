# Auditoria de cierre - seleccion inteligente de patches

## Alcance

Esta auditoria revisa la etapa actual de seleccion de patches:

```text
baseline_tiatoolbox
smart_tissue_nuclei_v1
smart_tissue_nuclei_v2_light
scripts/06_select_wsi_patches.py
scripts/07_compare_patch_selectors.py
src/selection/
outputs/patch_selection/
```

No se revisa segmentacion posterior, fine-tuning, HoVerNet, BCSS ground truth ni validacion clinica. No se cambiaron pesos, formulas de features, HED, cuotas espaciales, diversidad, numero de patches, candidate pool ni criterios de comparacion.

## Resumen ejecutivo

No se detectaron problemas criticos pendientes para cerrar la etapa actual.

El selector candidato final queda documentado como `smart_tissue_nuclei_v2_light`. `baseline_tiatoolbox` queda como baseline comparativo y `smart_tissue_nuclei_v1` como version intermedia/ablation.

## 1. Consistencia de nombres de selectores

**Estado:** OK.

- `baseline_tiatoolbox` se usa como baseline comparativo.
- `smart_tissue_nuclei_v1` se conserva como version intermedia/ablation.
- `smart_tissue_nuclei_v2_light` se documenta como selector propio candidato final.

**Hallazgo menor corregido:** el docstring de `run_smart_tissue_nuclei_selection` mencionaba solo v1 aunque la funcion soporta v1 y v2_light. Se ajusto el texto sin cambiar comportamiento.

## 2. Consistencia de columnas CSV

**Estado:** OK.

`candidate_metadata.csv` representa el pool comun thumbnail-filtered. `selected_metadata.csv` contiene solo seleccionados. En la comparacion, `comparison_selected_patches.csv` incluye:

- `nuclear_signal_recomputed`;
- `nuclear_signal_rgb_recomputed`;
- `nuclear_signal_hed_recomputed`.

`nuclear_signal_recomputed` queda como proxy RGB legacy por compatibilidad. HED aparece explicitamente para v2_light.

## 3. Consistencia de keys JSON

**Estado:** OK.

Los summaries reportan:

- selector;
- candidate pool;
- candidate metadata semantics;
- conteos generados/filtrados/evaluados/seleccionados;
- runtime;
- configuracion de proxy nuclear, cuotas y diversidad cuando aplica;
- warning clinico.

`comparison_summary.json` incluye tambien la ruta de `comparison_preview_selected_only.png` y notas sobre los proxies nucleares RGB/HED.

## 4. Compatibilidad baseline/v1/v2

**Estado:** OK.

`scripts/07_compare_patch_selectors.py` funciona con:

- baseline vs `smart_tissue_nuclei_v1`;
- baseline vs `smart_tissue_nuclei_v2_light`.

Los campos especificos de v2, como `quota_grid`, `regions_covered`, `nuclear_proxy` o `feature_diversity_bonus`, se tratan como descriptivos y no deberian romper corridas donde no existan.

## 5. Manejo de errores

**Estado:** OK con limitaciones esperadas.

El codigo falla explicitamente cuando faltan archivos requeridos de una corrida, cuando la carpeta de salida ya existe sin `--overwrite`, o cuando parametros principales tienen valores invalidos.

**Pendiente futuro:** agregar tests unitarios pequeños para validacion de manifests y comparacion sin depender de WSI real.

## 6. Reproducibilidad de comandos

**Estado:** OK.

Los comandos de baseline, v2_light y comparacion final estan documentados en:

- `README.md`;
- `docs/parte_2/plan_codigo.md`;
- `docs/parte_2/resultados_patch_selection.md`.

Usan `seed 42`, `patch_size 1024`, `stride 1024`, `max_patches 16` y el mismo pool comun de candidatos.

## 7. Riesgos de datos hardcodeados

**Clasificacion:** Menor.

Los comandos documentados usan una ruta local absoluta a la WSI de prueba:

```text
/Users/davidkripper/demoCasesMvpFeria/TCGA-A2-A3XS-01Z-00-DX1.867925C0-91D8-40A0-9FEA-25A635AC31E7.svs
```

Esto es correcto para reproducir la corrida local actual, pero no es portable a otro equipo sin ajustar rutas. No hay datos sensibles ni WSI copiados al repo.

## 8. Duplicacion innecesaria

**Estado:** Aceptable.

Hay duplicacion documental entre README, plan de codigo y resultados. Se mantiene intencionalmente porque cada documento tiene audiencia distinta:

- README: entrada rapida al repo;
- plan de codigo: bitacora tecnica de Parte II;
- resultados: explicacion didactica del pipeline y la comparacion final.

No se recomienda refactorizar documentacion en esta etapa.

## 9. Posibles errores silenciosos

**Clasificacion:** Menor.

`comparison_preview_selected_only.png` puede reconstruir un thumbnail limpio desde la WSI si el archivo existe. Si no puede abrir la WSI, cae a un thumbnail blanco con aviso visual. Esto mantiene compatibilidad, pero el preview selected-only seria menos informativo.

**Hallazgo menor corregido:** la descripcion CLI del comparador decia que no abria la WSI. Se ajusto para no prometer eso, porque ahora puede abrir solo el thumbnail para visualizacion. La comparacion de features sigue usando PNGs seleccionados.

## 10. Documentacion pendiente

**Pendiente futuro.**

- Evaluacion en mas WSIs.
- Evaluacion con ground truth BCSS para cobertura por clase.
- Segmentacion semantica posterior sobre patches seleccionados.
- Tests unitarios de manifests/comparacion.
- Definir convencion portable para rutas de datasets en servidores iHealth.

## Cambios menores aplicados durante esta auditoria

- README y plan de Parte II actualizados para congelar `smart_tissue_nuclei_v2_light` como selector candidato final.
- Comando principal de comparacion actualizado a baseline vs v2_light.
- Docstring de smart selector generalizado para v1/v2_light.
- Docstring de cuotas espaciales corregido de "cup redistribution" a "quota redistribution".
- Descripcion CLI del comparador ajustada para evitar una afirmacion obsoleta sobre abrir WSI.

Estos cambios son de documentacion/mensajes. No cambian la logica principal del selector ni de la comparacion.

## Conclusion

La etapa actual queda cerrada tecnicamente para entrega: artefactos finales copiados, selector candidato final documentado, comparacion baseline vs v2_light reproducible y sin hallazgos criticos pendientes.
