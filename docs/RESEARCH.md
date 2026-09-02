# Guía de investigación

## Alcance de la línea base

El directorio `dev/` contiene prototipos incrementales, no un paquete de producción. Cada subdirectorio tiene importaciones locales, por lo que los puntos de entrada deben ejecutarse desde su propia ruta tal como se indica en el README. Esta preparación no modifica la implementación ni las condiciones de sus experimentos.

## Orden recomendado de trabajo

1. Reproducir y registrar las baselines de bandits y Q-learning tabular.
2. Establecer una métrica común: recompensa media, desviación estándar y porcentaje de selección de la acción óptima.
3. Comparar DQN sin estado y con estado bajo el mismo presupuesto de interacciones.
4. Definir con precisión el objetivo de NoProp y compararlo contra el baseline DQN equivalente.
5. Añadir nuevas variantes fuera de los directorios históricos o con una configuración explícita por experimento.

## Reproducibilidad

Para cada corrida se debe registrar como mínimo:

- identificador de experimento y commit de Git;
- semilla de `random`, NumPy y PyTorch;
- entorno, distribución de recompensas y cambios de estado;
- hiperparámetros (episodios, `epsilon`, tasa de aprendizaje, buffer, lote y difusión);
- métricas por semilla y un resumen agregado;
- versión de Python y el `uv.lock` usado.

No compares una sola ejecución: las recompensas y la política epsilon-greedy son estocásticas. Usa múltiples semillas independientes y reporta media con dispersión.

## Validación sugerida antes de cambiar algoritmos

1. Verificar analíticamente la recompensa esperada de cada brazo.
2. Confirmar que la política aprendida supera a una política aleatoria bajo las mismas condiciones.
3. Asegurar que los datos de evaluación proceden de un entorno nuevo y que no se actualizan pesos durante ella.
4. Guardar figuras y tablas en `outputs/` o `runs/` (ambas rutas están ignoradas por Git).

## Observaciones de la auditoría

- Los archivos Python actuales son sintácticamente válidos.
- No existen pruebas automatizadas ni configuración de formato/lint aplicada al código histórico.
- Los scripts no establecen semillas, por lo que sus resultados no son deterministas.
- Los experimentos de bandits usan objetivos inmediatos (`target = reward`), coherentes con un bandit, mientras que la documentación DQN compartida describe el caso general con transición y descuento.
- Git contiene tanto `dev/4_NoPropDQL/DIffusion.py` como `dev/4_NoPropDQL/Diffusion.py`. Esto provoca una colisión de mayúsculas/minúsculas en sistemas de archivos de Windows y macOS habituales. No se modifica aquí para preservar la línea base; debe normalizarse en una futura tarea dedicada antes de integrar CI multiplataforma.
