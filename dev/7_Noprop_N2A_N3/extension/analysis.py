import csv
import json
from collections import defaultdict

import numpy as np
from neural.evaluation import value_metrics

from . import OLD, ROOT
from .evaluation import decisions, gap_summary
from .io import completed, config, digest, json_write, now, npz_write


def csv_write(name, rows):
    p = ROOT / "results/aggregated" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def paired(values, indices):
    values = np.asarray(values, float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite value in inference")
    ci = np.quantile(values[indices].mean(1), [0.025, 0.975])
    sd = float(values.std(ddof=1))
    return {
        "mean": float(values.mean()),
        "ci95_low": float(ci[0]),
        "ci95_high": float(ci[1]),
        "paired_sd": sd,
        "dz": float(values.mean() / sd) if sd else None,
        "fraction_negative": float((values < 0).mean()),
        "seeds": len(values),
    }


def contrast_rows(
    rows, pairs, metrics, sizes, seeds, families, bootstrap_seed, size_key="n"
):
    lookup = {(r[size_key], r["seed"], r["family"], r["method"]): r for r in rows}
    idx = np.random.default_rng(bootstrap_seed).integers(
        len(seeds), size=(10000, len(seeds))
    )
    result = []
    for size in sizes:
        for first, second in pairs:
            for metric in metrics:
                for family in ["all"] + list(families):
                    fs = families if family == "all" else [family]
                    difference = [
                        np.mean(
                            [
                                lookup[size, s, f, first][metric]
                                - lookup[size, s, f, second][metric]
                                for f in fs
                            ]
                        )
                        for s in seeds
                    ]
                    result.append(
                        {
                            size_key: size,
                            "first": first,
                            "second": second,
                            "metric": metric,
                            "family": family,
                            **paired(difference, idx),
                        }
                    )
    return result


def averages(rows, keys, metrics):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    return [
        dict(
            zip(keys, key),
            **{m: float(np.mean([r[m] for r in group])) for m in metrics},
            count=len(group),
        )
        for key, group in sorted(groups.items())
    ]


def analyze_n2a(require_complete=True):
    new = completed("N2A")
    if require_complete and len(new) != 480:
        raise RuntimeError(f"N2A incomplete: {len(new)}/480")
    spec = config("n2a")
    rows = []
    sources = {}
    paths = sorted((OLD / "results/raw/N2").glob("*/result.json")) + sorted(
        (ROOT / "results/raw/N2A").glob("*/result.json")
    )
    for p in paths:
        result = json.loads(p.read_text())
        job = result["config"]
        pp = p.parent / "predictions.npz"
        sources[str(pp)] = digest(pp)
        with np.load(pp) as saved:
            mu, rewards = saved["mu"], saved["rewards"]
            extracted = {}
            for key in ("value", "d1", "d2", "same_weights_average"):
                if key not in saved:
                    continue
                d = decisions(saved[key], mu)
                extracted.update({f"{key}_{k}": v for k, v in d.items()})
                stats = value_metrics(saved[key], mu, rewards)
                stats.pop("quantiles")
                stats.update(gap_summary(d["gap"], d["wrong_probability"]))
                q = stats.pop("wrong_gap_quantiles")
                stats.update({f"wrong_gap_{k}": v for k, v in q.items()})
                rows.append(
                    dict(
                        n=job["n"],
                        seed=job["seed"],
                        family=job["family"],
                        method=job["method"]
                        if key == "value"
                        else f"{job['method']}.{key}",
                        readout=key,
                        training_id=result["id"],
                        source_stage=job["stage"],
                        independently_trained=key == "value",
                        training_seconds=result["train_seconds"],
                        cost_note="full source training; no isolated readout training cost"
                        if key != "value"
                        else "observed training cost",
                        evaluation_seconds=result["evaluation_seconds"],
                        **stats,
                    )
                )
            npz_write(
                ROOT / "results/N2A_decisions" / f"{result['id']}.npz", **extracted
            )
    metrics = [
        "primary",
        "regret",
        "classification_error",
        "value_excess_bce",
        "value_excess_mse",
    ]
    pairs = []
    for stem in ("structured", "structured_mse"):
        pairs += [
            (stem, f"{stem}.{key}") for key in ("d1", "d2", "same_weights_average")
        ]
    pairs += [
        ("structured", "direct"),
        ("structured", "ensemble"),
        ("flexible", "ensemble"),
        ("structured_mse.d1", "structured.d1"),
        ("structured_mse", "structured"),
    ]
    if len(new) == 480:
        pairs += [
            ("structured_mse", "ensemble_mse"),
            ("ensemble_mse", "ensemble"),
            ("structured_mse.d1", "ensemble_mse"),
            ("structured_mse.same_weights_average", "ensemble_mse"),
        ]
    contrasts = contrast_rows(
        rows,
        pairs,
        metrics,
        spec["n"],
        spec["seeds"],
        spec["families"],
        spec["bootstrap_seed"],
    )
    summary = averages(
        rows, ["n", "method"], metrics + ["training_seconds", "evaluation_seconds"]
    )
    families = averages(rows, ["n", "family", "method"], metrics)
    csv_write("n2a_readouts.csv", rows)
    csv_write("n2a_contrasts.csv", contrasts)
    csv_write("n2a_summary.csv", summary)
    csv_write("n2a_families.csv", families)
    json_write(
        ROOT / "results/N2A_summary.json",
        {
            "utc": now(),
            "status": "completed" if len(new) == 480 else "historical_readouts_only",
            "new_fits": len(new),
            "reused_fits": 3360,
            "readout_rows": len(rows),
            "scope": "all exploratory; N2 unchanged",
            "direct_mse": "structured_mse.d1, same-training readout; equivalence tested at zero tolerance",
            "summary": summary,
            "contrasts": contrasts,
            "sources": sources,
        },
    )
    plots_n2a(summary, contrasts)
    report()
    return {"new_fits": len(new), "readout_rows": len(rows)}


def plots_n2a(summary, contrasts):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = ROOT / "results/figures"
    folder.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for method in (
        "structured_mse",
        "structured_mse.d1",
        "structured_mse.same_weights_average",
        "ensemble_mse",
        "ensemble",
        "flexible",
    ):
        selected = [r for r in summary if r["method"] == method]
        for ax, metric in zip(axes, ("primary", "regret")):
            ax.plot(
                [r["n"] for r in selected],
                [r[metric] for r in selected],
                "o-",
                label=method,
            )
            ax.set_xscale("log", base=2)
            ax.set_xlabel("n triples totales")
            ax.set_ylabel(metric)
            ax.grid(alpha=0.2)
    axes[1].legend(fontsize=7)
    fig.suptitle("N2A exploratorio: valor y decisiones, mismos contextos N2")
    fig.tight_layout()
    fig.savefig(folder / "n2a_value_decisions.png", dpi=180)
    plt.close(fig)


def freeze_n3(origin):
    spec = config("pilot_preregistered")
    results = completed("N3pilot")
    failures = list((ROOT / "results/raw/N3pilot").glob("*/failure.json"))
    failed_candidates = set()
    attempted_ids = {r["id"] for r in results}
    for path in failures:
        failure = json.loads(path.read_text())
        failed_candidates.add((failure["config"]["method"], failure["config"]["lr"]))
        attempted_ids.add(failure["id"])
    expected = len(spec["seeds"]) * len(spec["families"]) * len(spec["beta"])
    if len(attempted_ids) != expected * len(spec["methods"]) * len(
        spec["learning_rates"]
    ):
        raise RuntimeError("Pilot has unattempted runs; cannot freeze")
    cells = defaultdict(list)
    for r in results:
        if r["provenance"]["source_sha256"] != origin["source_sha256"]:
            raise RuntimeError("Pilot source mismatch")
        cells[r["config"]["method"], r["config"]["lr"]].append(r)
    rates, scores = {}, []
    for method in spec["methods"]:
        candidates = []
        for lr in spec["learning_rates"]:
            rr = cells[method, lr]
            valid = (
                (method, lr) not in failed_candidates
                and len(rr) == expected
                and all(
                    np.isfinite(r["metrics"]["observed_reward"])
                    and r["updates"] == 4999
                    for r in rr
                )
            )
            score = (
                float(np.mean([r["metrics"]["observed_reward"] for r in rr]))
                if valid
                else None
            )
            scores.append(
                {
                    "method": method,
                    "lr": lr,
                    "valid": valid,
                    "completed": len(rr),
                    "observed_reward": score,
                }
            )
            if valid:
                candidates.append((-score, lr))
        if not candidates:
            raise RuntimeError(f"No valid pilot rate for {method}")
        rates[method] = min(candidates)[1]
    # No oracle metrics available in these pilot results.
    budget = 4 * float(
        np.median([r["deployed_seconds"] for r in cells["direct", rates["direct"]]])
    )
    hashes = {
        r["id"]: digest(ROOT / "results/raw/N3pilot" / r["id"] / "result.json")
        for r in results
    }
    frozen = {
        "utc": now(),
        "status": "frozen_before_N3",
        "selected_rates": rates,
        "time_budget_seconds": budget,
        "scores": scores,
        "pilot_results": hashes,
        "pilot_failure_paths": [str(p.relative_to(ROOT)) for p in failures],
        "provenance": origin,
        "base": config("base"),
        "pilot": spec,
        "N3": config("n3_preregistered"),
        "hardware": config("hardware_selected"),
        "validation_sha256": digest(ROOT / "results/validation/latest.json"),
        "benchmark_sha256": digest(ROOT / "results/benchmark_cpu.json"),
    }
    target = ROOT / "configs/n3_frozen.json"
    if target.exists():
        previous = json.loads(target.read_text())
        for key in (
            "selected_rates",
            "time_budget_seconds",
            "pilot_results",
            "provenance",
        ):
            if previous[key] != frozen[key]:
                raise RuntimeError("Existing N3 freeze differs")
        return previous
    json_write(target, frozen)
    json_write(
        ROOT / "results/pilot_summary.json",
        {
            "utc": now(),
            "selected_rates": rates,
            "time_budget_seconds": budget,
            "scores": scores,
            "failures": len(failures),
            "completed": len(results),
        },
    )
    csv_write("pilot_scores.csv", scores)
    return frozen


def analyze_n3():
    results = completed("N3")
    if len(results) != 1440:
        raise RuntimeError(
            f"N3 incomplete {len(results)}/1440; no complete-case inference"
        )
    spec = config("n3_preregistered")
    rows, readout_rows = [], []
    for result in results:
        c = result["config"]
        if c["regime"] == "interactions" and (
            result["metrics"]["interactions"] != 20000 or result["updates"] != 19999
        ):
            raise RuntimeError("Incorrect primary checkpoint")
        r = dict(
            seed=c["seed"],
            family=c["family"],
            method=c["method"],
            regime=c["regime"],
            deployed_seconds=result["deployed_seconds"],
            diagnostic_seconds=result["clocks"]["diagnostic"],
            decision_seconds=result["clocks"]["decision"],
            update_seconds=result["clocks"]["update"],
            time_censored=result["time_censored"],
            **result["metrics"],
        )
        q = r.pop("wrong_gap_quantiles")
        r.update({f"wrong_gap_{k}": v for k, v in q.items()})
        rows.append(r)
        for diag in result["history"]:
            for readout, met in diag["metrics"].items():
                m = {k: v for k, v in met.items() if not isinstance(v, dict)}
                readout_rows.append(
                    dict(
                        seed=c["seed"],
                        family=c["family"],
                        regime=c["regime"],
                        method=c["method"]
                        + ("" if readout == "value" else "." + readout),
                        training_method=c["method"],
                        diagnostic_readout=readout,
                        deployed_readout=(
                            readout == "same_weights_average"
                            if c["method"] == "structured_average"
                            else readout == "value"
                        ),
                        t=diag["t"],
                        final=diag["t"] == result["metrics"]["interactions"],
                        **m,
                    )
                )
    pairs = [
        ("structured", "direct"),
        ("structured", "ensemble"),
        ("structured", "structured_average"),
        ("flexible", "ensemble"),
        ("flexible", "structured"),
    ]
    metrics = [
        "mean_regret",
        "wrong_fraction_non_tied",
        "observed_reward",
        "interactions",
        "deployed_seconds",
    ]
    contrasts = contrast_rows(
        rows,
        pairs,
        metrics,
        spec["regimes"],
        spec["seeds"],
        spec["families"],
        spec["bootstrap_seed"],
        "regime",
    )
    for row in contrasts:
        row["status"] = (
            "primary"
            if row["regime"] == "interactions"
            and row["first"] == "structured"
            and row["second"] == "direct"
            and row["metric"] == "mean_regret"
            and row["family"] == "all"
            else "secondary_exploratory"
        )
    attribution = []
    # Shared weights, fixed T: identical evaluation contexts and checkpoint for every pair.
    for t in spec["diagnostic_interactions"]:
        rr = [r for r in readout_rows if r["regime"] == "interactions" and r["t"] == t]
        pairs_same = [
            (m, m + "." + k)
            for m in ("structured", "structured_average", "flexible")
            for k in ("d1", "d2", "same_weights_average")
        ]
        attribution += contrast_rows(
            rr,
            pairs_same,
            ["primary", "regret", "classification_error"],
            [t],
            spec["seeds"],
            spec["families"],
            spec["bootstrap_seed"] + 1,
            "t",
        )
    summary = averages(
        rows,
        ["regime", "method"],
        metrics + ["decision_seconds", "update_seconds", "diagnostic_seconds"],
    )
    families = averages(rows, ["regime", "family", "method"], metrics)
    csv_write("n3_runs.csv", rows)
    csv_write("n3_contrasts.csv", contrasts)
    csv_write("n3_summary.csv", summary)
    csv_write("n3_families.csv", families)
    csv_write("n3_readouts.csv", readout_rows)
    csv_write("n3_same_checkpoint_contrasts.csv", attribution)
    json_write(
        ROOT / "results/N3_summary.json",
        {
            "utc": now(),
            "status": "completed",
            "fits": len(results),
            "summary": summary,
            "contrasts": contrasts,
            "same_checkpoint": attribution,
            "censored": sum(r["time_censored"] for r in rows),
        },
    )
    plots_n3(rows, readout_rows)
    report()


def plots_n3(rows, readouts):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, regime in zip(axes, ("interactions", "time")):
        for m in config("base")["methods"]:
            rr = [r for r in rows if r["regime"] == regime and r["method"] == m]
            ax.scatter(
                np.mean([r["deployed_seconds"] for r in rr]),
                np.mean([r["mean_regret"] for r in rr]),
                label=m,
            )
        ax.set_title(regime)
        ax.set_xlabel("Segundos desplegados")
        ax.set_ylabel("Regret / interacción")
        ax.grid(alpha=0.2)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    (ROOT / "results/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "results/figures/n3_cost_regret.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for m in config("base")["methods"]:
        for ax, metric in zip(axes, ("primary", "regret")):
            rr = [
                r
                for r in readouts
                if r["regime"] == "interactions"
                and r["training_method"] == m
                and r["deployed_readout"]
            ]
            ts = sorted({r["t"] for r in rr})
            ax.plot(
                ts,
                [np.mean([r[metric] for r in rr if r["t"] == t]) for t in ts],
                "o-",
                label=m,
            )
            ax.set_xscale("log")
            ax.set_xlabel("Interacciones")
            ax.set_ylabel(metric)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ROOT / "results/figures/n3_diagnostics.png", dpi=180)
    plt.close(fig)


def report():
    lines = [
        "# Extensión N2A y N3: informe experimental",
        "",
        f"Generado UTC: {now()}",
        "",
        "La campaña N0–N2 cerrada permanece intacta. N2A es una extensión exploratoria posterior: no modifica hipótesis, intervalos ni conclusiones confirmatorias originales.",
        "",
        "## N2A: pérdida, composición y decisiones",
        "",
    ]
    p = ROOT / "results/N2A_summary.json"
    if p.exists():
        d = json.loads(p.read_text())
        lookup = {(r["n"], r["method"]): r for r in d["summary"]}
        lines += [
            f"Estado: {d['status']}; {d['new_fits']}/480 ensembles MSE nuevos. Se reutilizan 3360 ajustes N2.",
            "",
            "El control directo/MSE es la lectura d1 del entrenamiento S/MSE: mismas capas, pérdida 0.5·media((sigmoid(logit)−r)²), SGD y actualizaciones bajo pesos/minibatches idénticos. No es una corrida independiente; su columna de coste conserva el coste completo del entrenamiento de origen. La tasa 0.1 del ensemble/MSE reproduce el protocolo previo y no se afirma óptima.",
            "",
            "| n | S/MSE | directo/MSE (d1) | promedio S/MSE | ensemble/MSE | F/BCE | ensemble/BCE |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
        methods = [
            "structured_mse",
            "structured_mse.d1",
            "structured_mse.same_weights_average",
            "ensemble_mse",
            "flexible",
            "ensemble",
        ]
        for n in (128, 512, 2048, 8192):
            lines.append(
                "|"
                + str(n)
                + "|"
                + "|".join(
                    f"{lookup[n, m]['primary']:.6f}"
                    if (n, m) in lookup
                    else "pendiente"
                    for m in methods
                )
                + "|"
            )
        lines += [
            "",
            "Error de valor: media sobre contextos del máximo error absoluto entre acciones. Contextos, particiones y checkpoints históricos comunes. Menor es mejor.",
            "",
            "Descomposición descriptiva por tamaño (no atribución causal porcentual):",
            "",
        ]
        for n in (128, 512, 2048, 8192):
            loss_change = (
                lookup[n, "structured_mse.d1"]["primary"]
                - lookup[n, "structured.d1"]["primary"]
            )
            composition_change = (
                lookup[n, "structured_mse"]["primary"]
                - lookup[n, "structured_mse.d1"]["primary"]
            )
            regret_change = (
                lookup[n, "structured_mse"]["regret"]
                - lookup[n, "structured_mse.d1"]["regret"]
            )
            lines.append(
                f"- n={n}: cambiar d1 de BCE a MSE cambia el error {loss_change:+.6f}; componer S/MSE frente a su d1 lo cambia {composition_change:+.6f} y cambia el regret {regret_change:+.6f}."
            )
        lines += [
            "",
            "BCE y MSE son entrenamientos históricos distintos con inicializaciones propias; la comparación anterior controla arquitectura, presupuesto y tasa, pero no afirma igualdad de pesos entre objetivos. Las diferencias de valor y de regret se contrastan por separado: un error funcional menor no garantiza mejores decisiones.",
            "",
            "| entrenamiento N2/N2A | segundos medios por ajuste |",
            "|---|---:|",
        ]
        for method in (
            "direct",
            "ensemble",
            "structured",
            "flexible",
            "structured_mse",
            "ensemble_mse",
        ):
            rr = [r for r in d["summary"] if r["method"] == method]
            if rr:
                lines.append(
                    f"|{method}|{np.mean([r['training_seconds'] for r in rr]):.3f}|"
                )
        lines += [
            "",
            "d1, p2 y promedio reutilizan el entrenamiento de origen; estas lecturas no tienen un coste de entrenamiento aislado medido. Los tiempos históricos y nuevos corresponden a sesiones distintas y no son una comparación controlada a igual reloj.",
            "",
            "| n | contraste exploratorio | métrica | diferencia | IC95% puntual |",
            "|---:|---|---|---:|---|",
        ]
        selected_pairs = [
            ("structured_mse", "structured_mse.d1"),
            ("structured_mse", "ensemble_mse"),
            ("structured_mse", "structured_mse.same_weights_average"),
            ("flexible", "ensemble"),
            ("structured_mse.d1", "structured.d1"),
        ]
        for r in d["contrasts"]:
            if (
                r["family"] == "all"
                and (r["first"], r["second"]) in selected_pairs
                and r["metric"] in ("primary", "regret")
            ):
                lines.append(
                    f"|{r['n']}|{r['first']} − {r['second']}|{r['metric']}|{r['mean']:+.6f}|[{r['ci95_low']:+.6f}, {r['ci95_high']:+.6f}]|"
                )
        lines += [
            "",
            "La comparación d1_MSE−d1_BCE separa la contribución del objetivo sin atribuirla a la composición. S/MSE−d1_MSE y S/MSE−promedio aíslan lecturas del mismo entrenamiento; el ensemble/MSE controla el promedio de dos redes entrenadas directamente. Estas comparaciones no identifican una tasa óptima ni una superioridad general de la cadena.",
            "",
            "Los archivos n2a_readouts.csv y N2A_decisions contienen elecciones, indicadores de empate, regret por contexto y distribuciones de brechas equivocadas. Las decisiones empatadas tienen coste esperado de desempate uniforme; argmax guardado es solo un diagnóstico. Los IC remuestrean 30 semillas completas, nunca contextos como réplicas.",
            "",
        ]
    else:
        lines += ["Análisis pendiente.", ""]
    lines += ["## N3: interacción y coste", ""]
    p = ROOT / "results/N3_summary.json"
    if p.exists():
        d = json.loads(p.read_text())
        lines += [
            f"Completadas {d['fits']}/1440 corridas (720 a 20000 interacciones y 720 a igual tiempo). Censuradas por límite operativo: {d['censored']}.",
            "",
            "| régimen | método | regret/interacción | decisiones erróneas | interacciones | segundos desplegados |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for r in d["summary"]:
            lines.append(
                f"|{r['regime']}|{r['method']}|{r['mean_regret']:.6f}|{r['wrong_fraction_non_tied']:.6f}|{r['interactions']:.1f}|{r['deployed_seconds']:.2f}|"
            )
        primary = next(r for r in d["contrasts"] if r["status"] == "primary")
        lines += [
            "",
            f"Contraste primario S−directo: {primary['mean']:+.6f}, IC95% [{primary['ci95_low']:+.6f}, {primary['ci95_high']:+.6f}], dz={primary['dz']}. Promedio de las cuatro familias dentro de semilla; 10000 remuestreos de semillas completas.",
            "",
            "Un intervalo que incluye cero no demuestra equivalencia. Ensemble, promedio, flexible, familias individuales e igualdad de tiempo son análisis secundarios exploratorios. La cuadratura se usa únicamente en diagnósticos fuera de entrenamiento y no entra en el reloj desplegado. Los tiempos de decisiones incluyen el promedio final del ensemble.",
            "",
        ]
        for regime in ("interactions", "time"):
            lines += [f"Contrastes secundarios ({regime}), regret/interacción:", ""]
            for r in d["contrasts"]:
                if (
                    r["regime"] == regime
                    and r["family"] == "all"
                    and r["metric"] == "mean_regret"
                    and r["status"] != "primary"
                ):
                    lines.append(
                        f"- {r['first']} − {r['second']}: {r['mean']:+.6f} [{r['ci95_low']:+.6f}, {r['ci95_high']:+.6f}]."
                    )
        structural = [
            r
            for r in d["contrasts"]
            if r["regime"] == "interactions"
            and r["metric"] == "mean_regret"
            and r["family"] == "all"
            and r["first"] == "structured"
        ]
        supported = all(r["ci95_high"] < 0 for r in structural)
        best_fixed = min(
            (r for r in d["summary"] if r["regime"] == "interactions"),
            key=lambda r: r["mean_regret"],
        )["method"]
        best_time = min(
            (r for r in d["summary"] if r["regime"] == "time"),
            key=lambda r: r["mean_regret"],
        )["method"]
        lines += [
            "",
            "## Decisión para S0–S1",
            "",
            f"La menor media observada de regret corresponde a {best_fixed} a igual T y a {best_time} a igual tiempo. Esta clasificación es descriptiva; no demuestra superioridad frente a todos los métodos ni sustituye los contrastes prefijados.",
            "",
            (
                "S/BCE compuesto merece pasar como candidato con controles directo, ensemble y promedio: todos los contrastes de regret previstos favorecen S. Los secundarios siguen siendo exploratorios."
                if supported
                else "La evidencia no establece una ventaja general de S/BCE compuesto sobre directo, ensemble y promedio a igual número de interacciones. S0–S1 debe estudiar la corrección categórica con directo y promedio como referencias, manteniendo S estructurado como candidato experimental; no se justifica avanzar a Atari por superioridad de composición."
            ),
            "",
            "S/MSE mostró una ventaja dependiente de pérdida y presupuesto en N2A; N3 no prueba su comportamiento interactivo porque MSE no forma parte de sus métodos congelados. Una extensión interactiva MSE requeriría su propio piloto.",
            "",
        ]
    else:
        lines += [
            "Pendiente de finalizar piloto, congelación y 1440 corridas confirmatorias/temporales. No se atribuyen aún resultados de interacción ni se elige una variante por rendimiento N3.",
            "",
        ]
    lines += [
        "## Reproducibilidad y límites",
        "",
        "Arquitecturas y tolerancias están en configs/base.json; registro previo en pilot_preregistered.json y n3_preregistered.json; tasas y presupuesto temporal en n3_frozen.json cuando exista. Cada corrida conserva configuración, fuente, buffer, estado ambiental, optimizadores y RNG separados. Reanudación exacta a igual interacción en el mismo entorno; el instante final de una corrida a igual tiempo depende de la carga y no es bit a bit reproducible.",
        "",
        "El plan S0–S1 y la referencia por programación dinámica están preparados por separado. S0–S1 no se ejecutan como campaña de aprendizaje. N4, S2–S3 y Atari permanecen pendientes; A3 no está autorizado por esta extensión.",
        "",
        "Tablas completas en results/aggregated; figuras en results/figures; evidencias de validación en results/validation; manifiesto de hashes en results/artifact_manifest.json. Los resultados por familia y dz no sustituyen la unidad experimental semilla.",
    ]
    (ROOT / "docs/REPORT.md").write_text("\n".join(lines) + "\n")
