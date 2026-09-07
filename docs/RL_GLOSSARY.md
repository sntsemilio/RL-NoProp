# Glosario y convención de nombres de RL

Este documento define el vocabulario utilizado en `dev/1_Stateless_ValueEstimation`,
`dev/2_Qlearn_Tabular` y `dev/3_DeepQLearn`. El prototipo
`dev/4_NoPropDQL` queda fuera de esta normalización hasta que se defina su
formulación definitiva.

## Reglas de nombres

- Variables y funciones usan `snake_case`.
- Clases usan `PascalCase`.
- Constantes de configuración usan `UPPER_SNAKE_CASE`.
- Una variable singular representa un elemento; una variable plural representa
  una colección o un lote.
- Se usa `action`, no `slot_id`, para la decisión tomada por el agente. Un
  `bandit_arm` es la distribución de recompensas asociada a esa acción.
- Se usa `training_step`, no `episode`, para una interacción aislada con un
  bandit. `episode` debe reservarse para una trayectoria que comienza con
  `reset()` y termina por terminación o truncamiento.
- Los tensores de un minibatch terminan en `_batch`; los registros tomados a lo
  largo del entrenamiento terminan en `_history`.

## Términos del entorno y del agente

| Concepto | Nombre en código | Significado |
| --- | --- | --- |
| Entorno | `environment` | Sistema con el que interactúa el agente. |
| Brazo del bandit | `bandit_arm` | Distribución de recompensas asociada a una acción. |
| Configuración de brazos | `BANDIT_ARMS` | Colección de todas las distribuciones de recompensa. |
| Estado | `state` | Información disponible antes de elegir una acción. Puede ser constante en un bandit no contextual. |
| Dimensión del estado | `state_dim` | Número de características de un vector de estado. |
| Acción | `action` | Índice de la alternativa elegida por la política. |
| Número de acciones | `num_actions` | Cantidad total de acciones disponibles. |
| Recompensa | `reward` | Señal escalar observada después de ejecutar una acción. |
| Resultados posibles | `reward_outcomes` | Valores que puede producir un brazo. |
| Probabilidades | `reward_probabilities` | Probabilidades correspondientes a `reward_outcomes`. |
| Política | `policy` | Regla que convierte un estado en una acción. |
| Política greedy | `greedy_policy` | Política que elige la acción con mayor Q-value estimado. |
| Exploración ε-greedy | `epsilon` | Probabilidad actual de seleccionar una acción al azar. |
| Límites de ε | `epsilon_start`, `epsilon_end` | Valores inicial y mínimo del programa de exploración. |
| Decaimiento de ε | `epsilon_decay` o `epsilon_decay_rate` | Factor multiplicativo o tasa exponencial que reduce `epsilon`. |

## Valores de acción y aprendizaje

El experimento `1_Stateless_ValueEstimation` aproxima la recompensa esperada
de cada acción mediante muestreo y actualización incremental sin estado
(stateless learning). Aquí `q_value` representa esa estimación; no implica
que el experimento implemente Q-learning.

| Concepto | Nombre en código | Significado |
| --- | --- | --- |
| Valor de acción | `q_value` | Estimación de retorno para una acción; en estos bandits es la recompensa inmediata esperada. |
| Valores de todas las acciones | `q_values` | Vector con un Q-value por acción. |
| Historial de Q-values | `q_value_history` | Estimaciones guardadas a lo largo del entrenamiento. |
| Número de muestras | `num_samples` | Cantidad de recompensas usadas para estimar el valor de una acción por separado. |
| Conteo de muestras | `sample_count` | Número de recompensas incorporadas a una estimación incremental. |
| Predicción seleccionada | `predicted_q_values` | Q-values que la red predijo para las acciones presentes en un minibatch. |
| Objetivo de aprendizaje | `q_targets` | Valores que deben aproximar las predicciones. En estos bandits son iguales a `reward_batch`. |
| Tamaño de paso tabular | `step_size` | Coeficiente α de una actualización incremental de Q. |
| Tamaño de paso constante | `constant_step_size` | Valor opcional de α que reemplaza el promedio muestral `1 / sample_count`. |
| Tasa del optimizador | `learning_rate` | Tasa usada por el optimizador para actualizar parámetros de una red neuronal. |
| Pérdida | `q_loss` | Error entre `predicted_q_values` y `q_targets`. |

`step_size` y `learning_rate` cumplen papeles relacionados, pero se mantienen
separados para distinguir una actualización tabular directa de una actualización
de parámetros mediante un optimizador.

## DQN y replay

| Concepto | Nombre en código | Significado |
| --- | --- | --- |
| Red Q online | `online_network` | Red entrenable que produce los Q-values usados por el agente. |
| Red Q objetivo | `target_network` | Copia sincronizada periódicamente para construir objetivos estables en DQN. |
| Intervalo de sincronización | `target_sync_interval` | Número de pasos entre copias de parámetros hacia la red objetivo. |
| Replay buffer | `replay_buffer` | Memoria finita de experiencias anteriores. |
| Transición | `transition` | Experiencia individual. En estos bandits contiene `(state, action, reward)`. |
| Minibatch de transiciones | `transition_batch` | Muestra aleatoria extraída del replay buffer. |
| Lotes | `state_batch`, `action_batch`, `reward_batch` | Tensores obtenidos al separar un minibatch por componente. |
| Tamaño de minibatch | `batch_size` | Número de transiciones usadas en una actualización. |

La `target_network` se conserva en los prototipos de DQN para mantener la
estructura del algoritmo general. En los experimentos actuales de bandits el
objetivo es solamente la recompensa inmediata, por lo que esa red todavía no
participa en el cálculo de `q_targets`.

## Pasos y métricas

| Concepto | Nombre en código | Significado |
| --- | --- | --- |
| Paso de entrenamiento | `training_step` | Una selección de acción, recompensa y posible actualización. |
| Presupuesto de entrenamiento | `num_training_steps` | Número total de interacciones de entrenamiento. |
| Pasos de evaluación | `num_evaluation_steps` | Interacciones realizadas sin aprendizaje para medir una política. |
| Recompensa media | `mean_reward` | Recompensa total dividida entre el número de pasos evaluados. |
| Acción óptima | `optimal_action` | Acción con mayor recompensa esperada para el estado actual. |
| Tasa de acción óptima | `optimal_action_rate` | Proporción de pasos en que la política eligió la acción óptima. |
| Duración del estado | `state_duration_steps` | Pasos durante los que el bandit contextual conserva el mismo estado. |

## Correspondencia matemática

| Notación | Nombre en código |
| --- | --- |
| \(s\) | `state` |
| \(a\) | `action` |
| \(r\) | `reward` |
| \(Q(s, a)\) | `q_value` |
| \(\alpha\) tabular | `step_size` |
| \(\varepsilon\) | `epsilon` |
| \(\theta\) | parámetros de `online_network` |
| \(\theta^-\) | parámetros de `target_network` |
