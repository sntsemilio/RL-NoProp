"""Reproducible tables, paired seed bootstrap, figures, and status-aware report."""

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-noprop-neural")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from neural.campaign import config, job_id, jobs
from neural.io import ROOT, json_write, now


def bootstrap(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(config("base")["analysis"]["bootstrap_seed"])
    idx = rng.integers(len(values), size=(10000, len(values)))
    means = values[idx].mean(1)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "n_seeds": len(values),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "ci9875_low": float(np.quantile(means, 0.00625)),
        "ci9875_high": float(np.quantile(means, 0.99375)),
    }


def csv_write(name, rows):
    if not rows:
        return
    p = ROOT / "results/aggregated" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def load(stage):
    return [
        json.loads(p.read_text())
        for p in sorted((ROOT / "results/raw" / stage).glob("*/result.json"))
    ]


def main():
    figures = ROOT / "results/figures"
    figures.mkdir(parents=True, exist_ok=True)
    n1, pilot, n2 = [load(s) for s in ["N1", "pilot", "N2"]]
    status = (
        json.loads((ROOT / "results/status.json").read_text())
        if (ROOT / "results/status.json").exists()
        else {}
    )
    n0 = json.loads((ROOT / "results/n0/latest.json").read_text())
    lines = [
        "# Informe experimental NoProp-RL neuronal v0.2",
        f"Generado UTC: {now()}.",
        "Este informe distingue pruebas científicas, aprendizaje exploratorio, selección reservada y confirmación. Las etapas incompletas no se interpretan como evidencia confirmatoria.",
        f"## Estado\nN0: **{n0['status']}**. N1: {len(n1)}/1980 ajustes. Piloto: {len(pilot)}/1680 ajustes. N2: {len(n2)}/3360 ajustes.",
        f"Proceso/etapa registrada: `{json.dumps(status, ensure_ascii=False)}`.",
        "E0/E1 históricos se preservaron; 156/156 artefactos históricos comprobados coincidieron con su manifiesto. E2 confirmatorio, E3, N3/N4, S0–S3 y A0–A4 permanecen pendientes. No se lanzó A3.",
        "## Especificación y procedencia",
        "Fuente principal: `dev/4_Noprop_fst_Advance/docs/NoProp-RL-Neuronal-Atari-v0.2.pdf`, SHA-256 `25af4522b187b1b7cfc621029bff47d2f441de568e94e5affcd86aee8972365c`. Fuente anterior: `avances_1.pdf`, SHA-256 `fee2bd5bde5cad11169dabad1151fcc5edcc39d9f6d4611e32aae1f378b2ceac`. Ambos documentos leídos completos; no se localizaron fuentes LaTeX ni verify_extensions.py. Base Git fc5bdd8; el PDF aportado sin versionar se preservó.",
        "`configs/sources.json`, `campaign_origin.json`, las configuraciones por corrida y los manifiestos registran documentos, versiones, commit y hashes de código con cambios locales. Los artefactos, checkpoints e informe numérico siguen la política local fuera de Git.",
        "## 1. Localidad, identidades y estabilidad",
        f"N0 ejecutó pytest con registro persistente: `{n0['log']}`. Tolerancias prefijadas en configs/n0.json, sin relajación. Las pruebas instrumentan llamadas prohibidas, pesos/optimizadores disjuntos, gradientes cruzados, diferencias finitas FP64 de todos los parámetros lejos de quiebres ReLU, igualdad de actualizaciones S1/directo, BCE estable, Bernoulli intermedia, RNG y reanudación bit a bit en CPU.",
        "La integración FP64 localiza todos los quiebres ReLU en z, divide los intervalos suaves y refina Gauss–Legendre 8→16→32→64→128 con tolerancia 1e-8. Se trunca cada gaussiana a ±10 desviaciones (masa exterior <1.6e-23). Se verifica contra una integración más precisa y Monte Carlo independiente sobre redes reales entrenadas. Las pruebas de identidad y cotas no constituyen resultados de generalización.",
        "El entrenamiento FP32 usa MLP 32–32/ReLU, acción one-hot, inicialización normal con escala sqrt(2/fan-in) en ocultas y sqrt(1/fan-in) en salida, SGD sin momentum, clipping ni penalización. Se generan ruidos frescos e independientes por bloque desde recompensas reales. La inferencia mantiene Bernoulli y promedia probabilidades finales.",
        "## 2. Aprendizaje y generalización",
    ]
    curves = []
    for r in n1:
        for h in r["history"]:
            if "metrics" in h:
                for read, m in h["metrics"].items():
                    curves.append(
                        {
                            "seed": r["config"]["seed"],
                            "instance": r["config"]["family"],
                            "N": r["config"]["n"],
                            "method": r["config"]["method"],
                            "readout": read,
                            "step": h["step"],
                            "train_seconds": h["train_seconds"],
                            "primary": m["primary"],
                        }
                    )
    csv_write("n1_curves.csv", curves)
    if curves:
        fig, axs = plt.subplots(1, 2, figsize=(11, 4))
        for method in [
            "structured",
            "flexible",
            "direct",
            "ensemble",
            "legacy_bce",
            "legacy_mse",
        ]:
            rows = [
                r for r in curves if r["method"] == method and r["readout"] == "value"
            ]
            points = []
            for step in [0, 500, 2000, 5000]:
                rr = [r for r in rows if r["step"] == step]
                if rr:
                    points.append(
                        (
                            step,
                            np.mean([r["train_seconds"] for r in rr]),
                            np.mean([r["primary"] for r in rr]),
                        )
                    )
            if points:
                x = np.array(points)
                axs[0].plot(x[:, 0], x[:, 2], "-o", label=method)
                axs[1].plot(x[:, 1], x[:, 2], "-o", label=method)
        for ax, xlabel in zip(axs, ["Actualizaciones", "Segundos de entrenamiento"]):
            ax.set(
                xlabel=xlabel,
                ylabel="Error máximo entre brazos",
                title=f"N1 exploratorio ({len(n1)}/1980 ajustes)",
            )
            ax.grid(alpha=0.2)
        axs[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / "n1_learning.png", dpi=150)
        plt.close(fig)
        rows = []
        for N in [128, 2048]:
            for method in sorted({r["config"]["method"] for r in n1}):
                subset = [
                    r
                    for r in n1
                    if r["config"]["n"] == N and r["config"]["method"] == method
                ]
                rows.append(
                    {
                        "N": N,
                        "method": method,
                        "completed": len(subset),
                        "expected": 90,
                        "primary_mean": float(
                            np.mean([r["metrics"]["value"]["primary"] for r in subset])
                        )
                        if subset
                        else None,
                        "tabular_mean": float(
                            np.mean([r["tabular_primary"] for r in subset])
                        )
                        if subset
                        else None,
                    }
                )
        csv_write("n1_summary.csv", rows)
        lines += [
            "N1 es una transición sin contexto, con N observaciones **por brazo**, nueve instancias y diez semillas nuevas. Las curvas incluyen 0, 500, 2000 y 5000 actualizaciones. No prueba necesidad de representación neuronal.",
            "![N1](../results/figures/n1_learning.png)",
            "| N | Método | Ajustes | Error medio | Tabular |",
            "|---:|---|---:|---:|---:|",
        ]
        for r in rows:
            if r["primary_mean"] is not None:
                lines.append(
                    f"|{r['N']}|{r['method']}|{r['completed']}/90|{r['primary_mean']:.6f}|{r['tabular_mean']:.6f}|"
                )
    else:
        lines.append("N1 todavía no tiene ajustes terminados.")
    if pilot:
        pr = []
        for m in config("base")["methods"]:
            for lr in [0.001, 0.01, 0.1]:
                rr = [
                    r
                    for r in pilot
                    if r["config"]["method"] == m and r["config"]["lr"] == lr
                ]
                if rr:
                    pr.append(
                        {
                            "method": m,
                            "lr": lr,
                            "completed": len(rr),
                            "expected": 80,
                            "validation_value_bce": float(
                                np.mean([r["validation_value_bce"] for r in rr])
                            ),
                        }
                    )
        csv_write("pilot_selection.csv", pr)
        lines += [
            "### Piloto reservado",
            "Selección por BCE de valor desde (c,a), con cuadratura para compuestos y recompensas independientes usadas solo en la pérdida de validación. Todas las familias reciben 3 tasas × 80 entornos × 5000 pasos. Se conserva el grid prefijado; una tasa elegida en un extremo limita la interpretación de optimalidad de la selección.",
            "| Método | Tasa | Ajustes | BCE validación |",
            "|---|---:|---:|---:|",
        ]
        for r in pr:
            lines.append(
                f"|{r['method']}|{r['lr']}|{r['completed']}/80|{r['validation_value_bce']:.6f}|"
            )
    if (ROOT / "configs/n2_frozen.json").exists():
        frozen = json.loads((ROOT / "configs/n2_frozen.json").read_text())
        lines.append(
            f"Protocolo congelado antes de N2: {frozen['utc']}. Tasas: `{json.dumps(frozen['selected_rates'])}`."
        )
    else:
        lines.append(
            "N2 todavía no está congelado. El supervisor lo congelará únicamente después de completar N1 y el piloto; todavía no se seleccionaron tasas confirmatorias."
        )
    lines += ["## 3. Utilidad de la composición y controles"]
    expected_n2 = set()
    if (ROOT / "configs/n2_frozen.json").exists():
        expected_n2 = {job_id(j) for j in jobs("N2", frozen["selected_rates"])}
    complete = bool(expected_n2) and {r["id"] for r in n2} == expected_n2
    flattened = []
    for r in n2:
        for read, m in r["metrics"].items():
            flattened.append(
                {
                    "id": r["id"],
                    "seed": r["config"]["seed"],
                    "family": r["config"]["family"],
                    "n": r["config"]["n"],
                    "method": r["config"]["method"],
                    "readout": read,
                    "trained_independently": read == "value",
                    "train_seconds_source_checkpoint": r["train_seconds"],
                    "evaluation_seconds": r["evaluation_seconds"],
                    "parameters_source_checkpoint": r["parameters"],
                    **{k: v for k, v in m.items() if k != "quantiles"},
                    **m["quantiles"],
                }
            )
    csv_write("n2_per_replica.csv", flattened)
    if complete:
        cells = defaultdict(dict)
        for row in flattened:
            key = (row["n"], row["method"], row["readout"], row["seed"])
            cells[key][row["family"]] = row["primary"]
        seedmeans = {
            k: np.mean(list(v.values())) for k, v in cells.items() if len(v) == 4
        }
        contrasts = []
        aggregates = []
        for n in [128, 512, 2048, 8192]:
            methods = sorted({(r["method"], r["readout"]) for r in flattened})
            seeds = config("n2")["seeds"]
            main = np.array([seedmeans[(n, "structured", "value", s)] for s in seeds])
            for method, read in methods:
                vals = np.array([seedmeans[(n, method, read, s)] for s in seeds])
                aggregates.append(
                    dict(n=n, method=method, readout=read, **bootstrap(vals))
                )
                if (method, read) != ("structured", "value"):
                    contrasts.append(
                        dict(
                            n=n,
                            comparison=f"structured/value minus {method}/{read}",
                            confirmatory=method == "direct" and read == "value",
                            **bootstrap(main - vals),
                        )
                    )
        csv_write("n2_aggregates.csv", aggregates)
        csv_write("n2_paired_contrasts.csv", contrasts)
        main = [r for r in contrasts if r["confirmatory"]]
        lines += [
            "Todos los 3360 ajustes están completos. El contraste usa la semilla completa: se promedian las cuatro familias dentro de cada semilla antes del bootstrap pareado de 10000 remuestreos. Signo negativo favorece NoProp-RL-S.",
            "| n total | S − directo | IC98.75 % Bonferroni | IC95 % puntual |",
            "|---:|---:|---|---|",
        ]
        for r in main:
            lines.append(
                f"|{r['n']}|{r['mean']:+.6f}|[{r['ci9875_low']:+.6f}, {r['ci9875_high']:+.6f}]|[{r['ci95_low']:+.6f}, {r['ci95_high']:+.6f}]|"
            )
        favorable = sum(r["ci9875_high"] < 0 for r in main)
        unfavorable = sum(r["ci9875_low"] > 0 for r in main)
        lines.append(
            f"Con ajuste por cuatro tamaños: {favorable} contrastes favorecen S, {unfavorable} favorecen el directo; {4 - favorable - unfavorable} incluyen cero. Incluir cero no demuestra equivalencia. Las comparaciones con ensemble, MLP ampliado y lecturas de los mismos pesos son exploratorias y están en n2_paired_contrasts.csv; una ventaja solo frente al logístico no identifica utilidad de composición."
        )
        lines += [
            "| n | S − control | Diferencia | IC95 % exploratorio |",
            "|---:|---|---:|---|",
        ]
        for r in contrasts:
            if r["comparison"].endswith(
                (
                    "ensemble/value",
                    "wide/value",
                    "structured/d1",
                    "structured/d2",
                    "structured/same_weights_average",
                )
            ):
                lines.append(
                    f"|{r['n']}|{r['comparison']}|{r['mean']:+.6f}|[{r['ci95_low']:+.6f}, {r['ci95_high']:+.6f}]|"
                )
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        for method in [
            "logistic",
            "direct",
            "wide",
            "ensemble",
            "structured",
            "flexible",
            "structured_mse",
        ]:
            rr = [
                r
                for r in aggregates
                if r["method"] == method and r["readout"] == "value"
            ]
            axs[0].plot(
                [r["n"] for r in rr], [r["mean"] for r in rr], "-o", label=method
            )
        axs[0].set(
            xscale="log", xlabel="n: triples totales", ylabel="Error de valor primario"
        )
        axs[0].legend(fontsize=7)
        axs[1].errorbar(
            range(4),
            [r["mean"] for r in main],
            yerr=[
                [r["mean"] - r["ci9875_low"] for r in main],
                [r["ci9875_high"] - r["mean"] for r in main],
            ],
            fmt="o",
            capsize=4,
        )
        axs[1].axhline(0, color="black", lw=1)
        axs[1].set(
            xticks=range(4),
            xticklabels=[r["n"] for r in main],
            xlabel="n",
            ylabel="S − directo; IC98.75 %",
        )
        fig.tight_layout()
        fig.savefig(figures / "n2_primary.png", dpi=160)
        plt.close(fig)
        lines.append("![N2](../results/figures/n2_primary.png)")
        # Heterogeneity is retained; never select favorable families.
        familyrows = []
        for n, f, m in [
            (n, f, m)
            for n in [128, 512, 2048, 8192]
            for f in config("n2")["families"]
            for m in config("base")["methods"]
        ]:
            rr = [
                r
                for r in flattened
                if r["n"] == n
                and r["family"] == f
                and r["method"] == m
                and r["readout"] == "value"
            ]
            familyrows.append(
                dict(n=n, family=f, method=m, **bootstrap([r["primary"] for r in rr]))
            )
        csv_write("n2_per_family.csv", familyrows)
    else:
        lines.append(
            "No se calculan contrastes confirmatorios con un subconjunto de semillas o métodos. La utilidad contextual de la composición permanece pendiente hasta completar N2; N1 no sustituye ese contraste."
        )
    lines += ["## 4. Coste computacional"]
    cpu = json.loads((ROOT / "results/benchmark_cpu.json").read_text())
    gpu = json.loads((ROOT / "results/benchmark_cuda.json").read_text())
    lines += [
        "CPU Ryzen 9 7940HS, 16 hilos lógicos; RAM WSL ~7.4 GiB. GPU RTX 4070 Laptop, 8 GiB, accesible fuera del sandbox. El benchmark incluye calentamiento y sincronización CUDA. Se eligió CPU, cuatro procesos y un hilo por proceso; memoria medida ~677 MiB por worker.",
        "| Método | CPU s/5000 | GPU s/5000 | CPU cuadratura s/8192 pares |",
        "|---|---:|---:|---:|",
    ]
    for row in cpu["rows"]:
        if row["threads"] != 1:
            continue
        gr = next(
            (r for r in gpu.get("rows", []) if r["method"] == row["method"]), None
        )
        lines.append(
            f"|{row['method']}|{row['train_seconds_5000']:.3f}|{gr['train_seconds_5000']:.3f}|{row['quadrature_seconds_per_8192_pairs']:.3f}|"
        )
    costs = []
    for stage, results in [("N1", n1), ("pilot", pilot), ("N2", n2)]:
        for method in config("n1")["methods"]:
            rr = [r for r in results if r["config"]["method"] == method]
            if rr:
                costs.append(
                    {
                        "stage": stage,
                        "method": method,
                        "runs": len(rr),
                        "train_seconds_mean": float(
                            np.mean([r["train_seconds"] for r in rr])
                        ),
                        "evaluation_seconds_mean": float(
                            np.mean([r["evaluation_seconds"] for r in rr])
                        ),
                    }
                )
    csv_write("costs.csv", costs)
    lines.append(
        "costs.csv separa entrenamiento y evaluación por familia. predictions.npz y result.json conservan lecturas, calibración, riesgos locales, correlación de errores y latencia. Las lecturas alternativas atribuyen el coste al checkpoint completo del que proceden, aunque su inferencia ejecute un solo bloque. Igual número de pasos no significa igual tiempo ni convergencia al mismo óptimo."
    )
    lines += [
        "## 5. Siguiente experimento y límites",
        "Si N1 muestra ajuste pero N2 no favorece la composición frente a directo y ensemble, la evidencia favorece estudiar la simplificación y el denoising auxiliar antes de un ensayo visual costoso. Si hay ventaja, N3 debe comprobar si se conserva en decisiones y a presupuesto temporal medido. N3/N4 requieren sus propias corridas; S0/S1 añaden objetivos categóricos y referencia secuencial conocida. Atari no queda validado por ningún resultado contextual.",
        "Las interfaces categóricas/proyección y el constructor CNN están preparados y probados solo como interfaces. No existen resultados de aprendizaje S0–S3 ni A0–A4. A3 permanece bloqueado y necesita un plan específico para sus 1200 millones de frames.",
        "## Reanudación y fallos",
        "Cada checkpoint incluye datos reales completos, pesos, estados de ambos optimizadores, estados RNG de minibatches y ruido, contador, historial, tiempos, configuración y procedencia. La reanudación exacta se verificó en el mismo entorno CPU; no se promete identidad entre dispositivos/versiones. Los procesos se reinician desde el último checkpoint atómico (cada 500 pasos), sin omitir semillas. Los logs de fallo son acumulativos; un fallo detiene la etapa y bloquea la siguiente. El manifiesto cubre artefactos locales; no constituye una copia externa de seguridad.",
        "Comandos de reproducción, consulta, parada y reanudación: README.md. El registro de decisiones fechado está en docs/DECISIONS.md.",
    ]
    failures = list((ROOT / "results/raw").glob("*/*/failure.json"))
    n0runs = list((ROOT / "results/n0").glob("*/summary.json"))
    lines.append(
        f"Fallos de corrida registrados: {len(failures)}. Ejecuciones N0 conservadas: {len(n0runs)}, fallidas: {sum(json.loads(p.read_text())['status'] != 'passed' for p in n0runs)}. No se eliminan para mejorar agregados."
    )
    for p in failures:
        lines.append(f"- `{p.relative_to(ROOT)}`")
    (ROOT / "docs/REPORT.md").write_text("\n\n".join(lines) + "\n")
    json_write(
        ROOT / "results/aggregated/analysis_status.json",
        {
            "utc": now(),
            "N1": len(n1),
            "pilot": len(pilot),
            "N2": len(n2),
            "N2_confirmatory_complete": complete,
        },
    )
    print("Report generated:", ROOT / "docs/REPORT.md")


if __name__ == "__main__":
    main()
