# RL-NoProp

Código y resultados de investigación sobre NoProp-RL, bandits contextuales y controles de aprendizaje directo. Estado actualizado al **14 de octubre de 2026**.

## Estado ejecutado

| Etapa | Estado | Ejecuciones terminadas |
| --- | --- | ---: |
| E0 / E1 | Completas | Controles matemáticos y experimento sin contexto |
| E2 / E3 | E2 solo smoke; confirmatorios pendientes | — |
| N0 | Validación completa | 28 pruebas |
| N1 | Exploratorio completo | 1 980 ajustes |
| Piloto N2 | Completo, tasas congeladas antes de N2 | 1 680 ajustes |
| N2 | Confirmatorio completo y auditado | 3 360 ajustes |
| N2A | Extensión exploratoria completa | 480 nuevos ensembles/MSE y lecturas de N2 |
| Piloto N3 | Completo, tasas congeladas antes de N3 | 1 920 trayectorias |
| N3 | Confirmatorio completo y auditado | 720 a 20 000 interacciones + 720 a igual tiempo |
| S0–S1 | Implementación de referencia y configuración preparadas | Sin entrenamiento |
| N4, S2–S3, Atari A0–A4 | Pendientes | Sin ejecución |

La extensión terminó el **14 de octubre de 2026 a las 01:44:57 UTC**. Los artefactos originales se verificaron después del reinicio de la máquina. El [estado numérico](published_results/status.json) y las [auditorías de N2A/N3](published_results/N2A_N3/audits.json) acompañan los resultados.

## Resultados principales

N2 compara el error de valor de S/BCE compuesto menos MLP directo/BCE, promediando las cuatro familias dentro de cada una de las 30 semillas. Un signo negativo favorece a S. Los intervalos del 98.75 % corresponden al ajuste Bonferroni de los cuatro tamaños; también se publican los intervalos puntuales del 95 %.

| Triples totales `n` | Diferencia media | IC 98.75 % |
| ---: | ---: | --- |
| 128 | −0.012728 | [−0.021134, −0.004299] |
| 512 | −0.006176 | [−0.010685, −0.001751] |
| 2048 | −0.003547 | [−0.005472, −0.001698] |
| 8192 | +0.000184 | [−0.001419, +0.001628] |

Estos contrastes no demuestran una ventaja general frente al ensemble ni frente al promedio de las propensiones. N2A es exploratorio: el primer bloque de S/MSE funciona como control directo/MSE emparejado porque se verificó la equivalencia de sus actualizaciones. Es una lectura del mismo entrenamiento, sin coste de entrenamiento aislado. El ensemble/MSE sí se entrenó por separado, conservando la tasa 0.1 del protocolo anterior.

En N3, **S/BCE compuesto tuvo mayor regret que el directo/BCE**. El contraste principal a 20 000 interacciones fue **+0.001512 por interacción**, IC 95 % **[+0.001332, +0.001697]**, con bootstrap pareado de 10 000 remuestreos de 30 semillas completas. Los tiempos siguientes excluyen cuadratura, diagnósticos y escritura de checkpoints.

| Agente N3 | Regret medio por interacción | Segundos desplegados por corrida |
| --- | ---: | ---: |
| Directo/BCE | 0.006941 | 25.38 |
| Ensemble/BCE | 0.006898 | 41.31 |
| S/BCE compuesto | 0.008453 | 47.69 |
| F/BCE compuesto | 0.007670 | 46.82 |
| Logístico contextual | 0.069183 | 20.28 |
| S/BCE, promedio de propensiones | 0.006821 | 42.86 |

La comparación secundaria a igual tiempo utilizó **24.707171 segundos**, fijados exclusivamente desde el piloto. S menos directo dio +0.004400, IC 95 % [+0.004114, +0.004709]. Las comparaciones secundarias son exploratorias; estas cifras no constituyen evidencia de desempeño en Atari.

## Código y resultados

| Ruta | Contenido |
| --- | --- |
| `dev/1_Stateless_ValueEstimation`, `dev/2_Qlearn_Tabular`, `dev/3_DeepQLearn` | Controles y prototipos iniciales |
| `dev/4_Noprop_fst_Advance` | Implementación y pruebas E0–E3 |
| `dev/5_NoPropDQL` | Prototipo histórico; no define la formulación neuronal actual |
| `dev/6_Noprop_Neural` | Bloques neuronales locales, cuadratura, controles y campaña N0–N2 |
| `dev/7_Noprop_N2A_N3` | Atribución N2A, agentes interactivos N3 y referencia MDP de S0–S1 |
| [Resultados E](published_results/E) | Tablas, métricas por corrida y figuras históricas |
| [Resultados N0–N2](published_results/N) | Curvas N1, métricas por réplica, contrastes y protocolo N2 resuelto |
| [Resultados N2A–N3](published_results/N2A_N3) | Lecturas, decisiones, costes, contrastes por familia y protocolo N3 resuelto |
| [Manifiesto público](published_results/manifest.json) | Tamaños y SHA-256 de los artefactos exportados |

Las configuraciones resueltas publicadas son extractos de los originales congelados: conservan arquitectura, semillas, tasas, presupuestos y análisis, y registran el hash del original. Los estados incluidos en el prerregistro se conservan tal como estaban antes de ejecutar; el estado actual está en `published_results/status.json`.

## Verificación reproducible

Requisitos: Python 3.11 o 3.12 y `uv`.

```bash
uv sync --locked --all-groups
uv run --locked python scripts/verify_public_results.py
uv run --locked pytest -q dev/4_Noprop_fst_Advance/tests dev/6_Noprop_Neural/tests dev/7_Noprop_N2A_N3/tests -k 'not offline_control_reuses_actual_historical_data'
```

La verificación reproduce los contrastes principales desde tablas por semilla y comprueba los hashes sin necesitar documentos ni checkpoints. La prueba excluida comprueba la reutilización de checkpoints históricos privados; permanece disponible para ejecutarse en el entorno original. Los supervisores de campañas y la reanudación exacta también requieren esos artefactos locales y su procedencia original.

La exportación se regenera en el entorno original con `python scripts/export_public_results.py`. Antes de publicar, `python scripts/check_publication.py --base origin/master` comprueba el índice y los commits nuevos. El repositorio publica código, configuraciones y resultados revisados. Los PDF, fuentes LaTeX, informes, notas, notebooks, checkpoints y registros de la máquina se conservan fuera del versionado público; este README es el índice técnico.
