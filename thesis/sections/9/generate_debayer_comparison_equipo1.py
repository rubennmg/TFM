from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Sequence, TypeAlias, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT.parent.parent.parent / "results" / "Equipo1"
OUTPUT_DIR = ROOT / "generated" / "debayer_equipo1"
FIG_DIR = OUTPUT_DIR / "figures"
TAB_DIR = OUTPUT_DIR / "tables"

PipelineKey: TypeAlias = Literal[
    "debayer2x2_run",
    "debayer3x3_run",
    "debayer5x5_run",
    "debayerSplit_run",
]
DebayerData: TypeAlias = dict[PipelineKey, dict[str, float]]
PipelineLatency: TypeAlias = tuple[PipelineKey, float]

DEBAYER_2X2: PipelineKey = "debayer2x2_run"
DEBAYER_3X3: PipelineKey = "debayer3x3_run"
DEBAYER_5X5: PipelineKey = "debayer5x5_run"
DEBAYER_SPLIT: PipelineKey = "debayerSplit_run"

PIPELINE_ORDER: tuple[PipelineKey, ...] = (
    DEBAYER_2X2,
    DEBAYER_3X3,
    DEBAYER_5X5,
    DEBAYER_SPLIT,
)

PIPELINE_LABELS: dict[PipelineKey, str] = {
    DEBAYER_2X2: "Debayer2x2",
    DEBAYER_3X3: "Debayer3x3",
    DEBAYER_5X5: "Debayer5x5",
    DEBAYER_SPLIT: "DebayerSplit",
}

RUN_FILES: dict[PipelineKey, str] = {
    DEBAYER_2X2: "0002_debayer2x2_run.json",
    DEBAYER_3X3: "0003_debayer3x3_run.json",
    DEBAYER_5X5: "0004_debayer5x5_run.json",
    DEBAYER_SPLIT: "0005_debayerSplit_run.json",
}

DISPLAY_ORDER: tuple[PipelineKey, ...] = (
    DEBAYER_2X2,
    DEBAYER_SPLIT,
    DEBAYER_3X3,
    DEBAYER_5X5,
)

PLOT_COLORS: dict[PipelineKey, str] = {
    DEBAYER_2X2: "#1f4e79",
    DEBAYER_SPLIT: "#2a9d8f",
    DEBAYER_3X3: "#e9c46a",
    DEBAYER_5X5: "#e76f51",
}

RESOLUTION_ORDER: list[str] = ["256x256", "512x512", "1024x1024", "2048x2048"]
LATEX_NL = "\\\\"


def _ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)


def _shape_label(shape: Sequence[int]) -> str:
    return f"{shape[1]}x{shape[2]}"


def _resolution_mpix(resolution: str) -> float:
    width, height = (int(value) for value in resolution.split("x"))
    return (width * height) / 1_000_000.0


def _run_file(pipeline: PipelineKey) -> Path:
    return RESULTS_DIR / RUN_FILES[pipeline]


def _load_gpu_float32() -> DebayerData:
    data: DebayerData = {pipeline: {} for pipeline in PIPELINE_ORDER}

    for pipeline in PIPELINE_ORDER:
        payload = cast(dict[str, Any], json.loads(_run_file(pipeline).read_text()))
        benchmarks = payload.get("benchmarks", [])
        if not isinstance(benchmarks, list):
            continue

        for bench in benchmarks:
            if not isinstance(bench, dict):
                continue

            params = bench.get("params", {})
            if not isinstance(params, dict):
                continue
            device_raw = str(params.get("device", "")).lower()
            if device_raw != "cuda":
                continue

            dtype_raw = str(params.get("dtype", ""))
            if "float32" not in dtype_raw:
                continue

            shape_raw = params.get("shape", [1, 0, 0])
            if not isinstance(shape_raw, list) or len(shape_raw) != 3:
                continue
            if not all(isinstance(item, int) for item in shape_raw):
                continue

            resolution = _shape_label(cast(Sequence[int], shape_raw))
            if resolution not in RESOLUTION_ORDER:
                continue

            stats = bench.get("stats", {})
            if not isinstance(stats, dict):
                continue

            median = stats.get("median")
            if not isinstance(median, int | float):
                continue

            data[pipeline][resolution] = float(median)

    return data


def _write_latency_fps_table(data: DebayerData) -> None:
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\renewcommand{\\arraystretch}{1.2}",
        "\\setlength{\\tabcolsep}{6pt}",
        "\\begin{tabular}{|c|r|r|r|r|}",
        "\\hline",
        "\\rowcolor[gray]{0.90}",
        "\\textbf{Resolución} & \\textbf{Debayer2x2 (ms)} & \\textbf{DebayerSplit (ms)} & \\textbf{Debayer3x3 (ms)} & \\textbf{Debayer5x5 (ms)} "
        + LATEX_NL,
        "\\hline",
    ]

    for resolution in RESOLUTION_ORDER:
        row = [
            data[DEBAYER_2X2][resolution] * 1000.0,
            data[DEBAYER_SPLIT][resolution] * 1000.0,
            data[DEBAYER_3X3][resolution] * 1000.0,
            data[DEBAYER_5X5][resolution] * 1000.0,
        ]
        lines.append(
            f"{resolution} & {row[0]:.3f} & {row[1]:.3f} & {row[2]:.3f} & {row[3]:.3f} {LATEX_NL}"
        )
        lines.append("\\hline")

    lines.extend(
        [
            "\\end{tabular}",
            "\\caption{Latencia mediana de módulos de \\textit{Debayer}}",
            "\\label{tab:debayer_gpu_latency_equipo1_float32}",
            "\\end{table}",
            "",
        ]
    )

    (TAB_DIR / "tab_debayer_gpu_latency_equipo1_float32.tex").write_text(
        "\n".join(lines)
    )


def _write_ranking_2048_table(data: DebayerData) -> None:
    resolution = "2048x2048"
    ranking: list[PipelineLatency] = sorted(
        [(pipeline, data[pipeline][resolution]) for pipeline in PIPELINE_ORDER],
        key=lambda item: item[1],
    )
    best = ranking[0][1]

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\renewcommand{\\arraystretch}{1.2}",
        "\\setlength{\\tabcolsep}{8pt}",
        "\\begin{tabular}{|c|l|r|r|}",
        "\\hline",
        "\\rowcolor[gray]{0.90}",
        "\\textbf{Posición} & \\textbf{Módulo Debayer} & \\textbf{Latencia (ms)} & \\textbf{Ratio vs mejor} "
        + LATEX_NL,
        "\\hline",
    ]

    for pos, (pipeline, value_s) in enumerate(ranking, start=1):
        ratio = value_s / best
        lines.append(
            f"{pos} & {PIPELINE_LABELS[pipeline]} & {value_s * 1000.0:.3f} & {ratio:.2f}x {LATEX_NL}"
        )
        lines.append("\\hline")

    lines.extend(
        [
            "\\end{tabular}",
            "\\caption{Clasificación de módulos de \\textit{Debayer} en \\texttt{2048x2048}}",
            "\\label{tab:debayer_gpu_ranking_2048_equipo1_float32}",
            "\\end{table}",
            "",
        ]
    )

    (TAB_DIR / "tab_debayer_gpu_ranking_2048_equipo1_float32.tex").write_text(
        "\n".join(lines)
    )


def _write_relative_table(data: DebayerData) -> None:
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\renewcommand{\\arraystretch}{1.2}",
        "\\setlength{\\tabcolsep}{7pt}",
        "\\begin{tabular}{|c|r|r|r|r|}",
        "\\hline",
        "\\rowcolor[gray]{0.90}",
        "\\textbf{Resolución} & \\textbf{Debayer2x2} & \\textbf{DebayerSplit} & \\textbf{Debayer3x3} & \\textbf{Debayer5x5} "
        + LATEX_NL,
        "\\hline",
    ]

    for resolution in RESOLUTION_ORDER:
        best = min(data[pipeline][resolution] for pipeline in PIPELINE_ORDER)
        ratios = [data[pipeline][resolution] / best for pipeline in PIPELINE_ORDER]
        lines.append(
            f"{resolution} & {ratios[0]:.2f}x & {ratios[3]:.2f}x & {ratios[1]:.2f}x & {ratios[2]:.2f}x {LATEX_NL}"
        )
        lines.append("\\hline")

    lines.extend(
        [
            "\\end{tabular}",
            "\\caption{Comparativa relativa de módulos \\textit{Debayer} respecto a \\texttt{Debayer2x2}}",
            "\\label{tab:debayer_gpu_relative_vs_best_equipo1_float32}",
            "\\end{table}",
            "",
        ]
    )

    (TAB_DIR / "tab_debayer_gpu_relative_vs_best_equipo1_float32.tex").write_text(
        "\n".join(lines)
    )


def _write_2048_mpix_cost_table(data: DebayerData) -> None:
    resolution = "2048x2048"
    mpix = _resolution_mpix(resolution)
    baseline = data[DEBAYER_2X2][resolution]

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\renewcommand{\\arraystretch}{1.2}",
        "\\setlength{\\tabcolsep}{6pt}",
        "\\begin{tabular}{|l|r|r|r|r|}",
        "\\hline",
        "\\rowcolor[gray]{0.90}",
        "\\textbf{Módulo} & \\textbf{Latencia (ms)} & \\textbf{Diferencia vs 2x2 (ms)} & \\textbf{Relación vs 2x2} & \\textbf{Coste ms/MPix} "
        + LATEX_NL,
        "\\hline",
    ]

    for pipeline in DISPLAY_ORDER:
        latency_s = data[pipeline][resolution]
        latency_ms = latency_s * 1000.0
        delta_ms = (latency_s - baseline) * 1000.0
        ratio = latency_s / baseline
        ms_per_mpix = latency_ms / mpix
        delta = f"+{delta_ms:.3f}" if delta_ms > 0 else f"{delta_ms:.3f}"
        lines.append(
            f"\\texttt{{{PIPELINE_LABELS[pipeline]}}} & {latency_ms:.3f} & {delta} & {ratio:.2f}x & {ms_per_mpix:.2f} {LATEX_NL}"
        )
        lines.append("\\hline")

    lines.extend(
        [
            "\\end{tabular}",
            "\\caption{Coste relativo y normalizado de los módulos de \\textit{Debayer} en \\texttt{2048x2048}}",
            "\\label{tab:debayer_2048_mpix_cost_equipo1_float32}",
            "\\end{table}",
            "",
        ]
    )

    (TAB_DIR / "tab_debayer_2048_mpix_cost_equipo1_float32.tex").write_text(
        "\n".join(lines)
    )


def _plot_latency_vs_resolution(data: DebayerData) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.2, 4.8))

    x = list(range(len(RESOLUTION_ORDER)))
    for pipeline in DISPLAY_ORDER:
        y = [data[pipeline][resolution] * 1000.0 for resolution in RESOLUTION_ORDER]
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.2,
            color=PLOT_COLORS[pipeline],
            label=PIPELINE_LABELS[pipeline],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(RESOLUTION_ORDER)
    ax.set_xlabel("Resolución")
    ax.set_ylabel("Latencia mediana (ms)")
    ax.legend(loc="upper left", frameon=True)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_debayer_gpu_latency_vs_resolution_equipo1_float32.pdf")
    plt.close(fig)


def _plot_ranking_2048(data: DebayerData) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.8, 4.6))

    resolution = "2048x2048"
    ranking: list[PipelineLatency] = sorted(
        [
            (pipeline, data[pipeline][resolution] * 1000.0)
            for pipeline in PIPELINE_ORDER
        ],
        key=lambda item: item[1],
    )

    labels = [PIPELINE_LABELS[pipeline] for pipeline, _ in ranking]
    values = [value for _, value in ranking]
    y = list(range(len(labels)))
    colors = [PLOT_COLORS[pipeline] for pipeline, _ in ranking]

    ax.barh(y, values, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Latencia mediana (ms)")

    for idx, value in enumerate(values):
        ax.text(value + 0.02, idx, f"{value:.3f} ms", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_debayer_gpu_ranking_2048_equipo1_float32.pdf")
    plt.close(fig)


def _plot_relative_vs_best(data: DebayerData) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.2, 4.8))

    x = list(range(len(RESOLUTION_ORDER)))
    for pipeline in DISPLAY_ORDER[1:]:
        ratios: list[float] = []
        for resolution in RESOLUTION_ORDER:
            best = min(data[p][resolution] for p in PIPELINE_ORDER)
            ratios.append(data[pipeline][resolution] / best)
        ax.plot(
            x,
            ratios,
            marker="o",
            linewidth=2.2,
            color=PLOT_COLORS[pipeline],
            label=PIPELINE_LABELS[pipeline],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(RESOLUTION_ORDER)
    ax.set_xlabel("Resolución")
    ax.set_ylabel("Ratio relativo al mejor módulo")
    ax.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_debayer_gpu_relative_vs_best_equipo1_float32.pdf")
    plt.close(fig)


def main() -> None:
    _ensure_dirs()
    data = _load_gpu_float32()

    _write_latency_fps_table(data)
    _write_ranking_2048_table(data)
    _write_relative_table(data)
    _write_2048_mpix_cost_table(data)

    _plot_latency_vs_resolution(data)
    _plot_ranking_2048(data)
    _plot_relative_vs_best(data)

    print(f"Assets generated in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
