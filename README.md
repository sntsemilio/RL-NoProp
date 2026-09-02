# RL-NoProp

Repositorio de investigación para estudiar agentes de aprendizaje por refuerzo en problemas de *multi-armed bandits*. La progresión va desde estimación de valor y Q-learning tabular hasta DQN y un prototipo de aprendizaje por difusión (NoProp).

El código existente se conserva como línea base: esta preparación no altera algoritmos, hiperparámetros ni resultados de los experimentos.

## Inicio rápido

Requisitos: [uv](https://docs.astral.sh/uv/) y Python 3.11 o 3.12. `uv` crea un entorno aislado en `.venv` y resuelve versiones reproducibles a partir de `uv.lock`.

```powershell
uv sync --all-groups
```

No es necesario activar el entorno para ejecutar los experimentos:

```powershell
uv run python dev/1_QLearn_Bandits/main.py
uv run python dev/2_Qlearn_Tabular/main.py
uv run python dev/3_DeepQLearn/v1_NoStates/main.py
uv run python dev/3_DeepQLearn/v2_States/main.py
uv run python dev/4_NoPropDQL/main.py
```

Los scripts de entrenamiento abren gráficos de Matplotlib y algunos pueden tardar por el número de episodios configurado. Para trabajar de forma interactiva:

```powershell
uv run jupyter lab
```

Para activar el entorno manualmente en PowerShell, usa el de este proyecto (no uno de otro repositorio):

```powershell
.\.venv\Scripts\Activate.ps1
```

## Mapa del proyecto

| Ruta | Propósito |
| --- | --- |
| `dev/1_QLearn_Bandits` | Estimación incremental de la recompensa esperada de dos bandits. |
| `dev/2_Qlearn_Tabular` | Baseline de Q-learning tabular con exploración epsilon-greedy. |
| `dev/3_DeepQLearn/v1_NoStates` | DQN para un bandit sin estado informativo. |
| `dev/3_DeepQLearn/v2_States` | DQN con estado contextual alternante (`AB`/`BA`). |
| `dev/4_NoPropDQL` | Prototipo NoProp basado en una cadena de difusión y MLPs de denoising. |
| `docs/RESEARCH.md` | Protocolo de experimentación, comparabilidad y observaciones de la auditoría. |

## Dependencias

- `numpy`: cálculo numérico.
- `matplotlib`: visualización de las curvas de aprendizaje.
- `torch`: redes neuronales y optimización para DQN/NoProp.
- Herramientas de desarrollo: JupyterLab, ipykernel, pytest y Ruff.

## Protocolo mínimo de investigación

Antes de comparar métodos, define una semilla, conserva la configuración ejecutada, mide varios *seeds* y guarda métricas agregadas. El código actual no fija semillas ni persiste resultados automáticamente, por lo que debe tratarse como una línea base exploratoria. Consulta [la guía de investigación](docs/RESEARCH.md) antes de abrir una nueva ronda experimental.

## Estado actual

Todos los módulos Python versionados superan compilación sintáctica. El primer `uv sync --all-groups` instalará las dependencias y generará el entorno local. Hay una colisión histórica de nombre (`DIffusion.py` y `Diffusion.py`) en el índice de Git; en Windows ambos nombres no pueden coexistir de forma fiable. Está documentada para resolverla antes de tocar ese experimento en un entorno Linux/CI.
