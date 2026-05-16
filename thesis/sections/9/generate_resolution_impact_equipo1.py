from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT.parent.parent.parent / "results" / "Equipo1"
OUTPUT_DIR = ROOT / "generated" / "resolution_equipo1"
FIG_DIR = OUTPUT_DIR / "figures"
TAB_DIR = OUTPUT_DIR / "tables"

PipelineKey = Literal["standard_run", "complex_run"]
DeviceKey = Literal["cpu", "cuda"]
DTypeKey = Literal["float16", "float32", "float64"]

PIPELINES: dict[PipelineKey, str] = {
    "standard_run": "standard",
    "complex_run": "complex",
}
PIPELINE_ORDER: list[PipelineKey] = ["standard_run", "complex_run"]
RESOLUTION_ORDER: list[str] = ["256x256", "512x512", "1024x1024", "2048x2048"]
DTYPE_ORDER: list[DTypeKey] = ["float16", "float32", "float64"]
DEVICES: list[DeviceKey] = ["cuda"]


def _device_label(device: DeviceKey) -> str:
    return "GPU" if device == "cuda" else "CPU"


def _ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)


def _run_file(pipeline: PipelineKey) -> Path:
    mapping = {
        "standard_run": "0006_standard_run.json",
        "complex_run": "0001_complex_run.json",
    }
    return RESULTS_DIR / mapping[pipeline]


def _shape_label(shape: Sequence[int]) -> str:
    return f"{shape[1]}x{shape[2]}"


def _resolution_mpix(resolution: str) -> float:
    width, height = (int(value) for value in resolution.split("x"))
    return (width * height) / 1_000_000.0


def _dtype_label(raw_dtype: str) -> DTypeKey | None:
    if "float16" in raw_dtype:
        return "float16"
    if "float32" in raw_dtype:
        return "float32"
    if "float64" in raw_dtype:
        return "float64"
    return None


def _load_records() -> dict[
    PipelineKey, dict[DeviceKey, dict[DTypeKey, dict[str, float]]]
]:
    # data[pipeline][device][dtype][resolution] = median_s
    data: dict[PipelineKey, dict[DeviceKey, dict[DTypeKey, dict[str, float]]]] = {}

    for pipeline in PIPELINE_ORDER:
        data[pipeline] = {"cpu": {}, "cuda": {}}
        payload = json.loads(_run_file(pipeline).read_text())

        for bench in payload.get("benchmarks", []):
            params = bench.get("params", {})
            device_raw = str(params.get("device", "")).lower()
            if device_raw not in {"cpu", "cuda"}:
                continue
            device: DeviceKey = "cpu" if device_raw == "cpu" else "cuda"

            dtype = _dtype_label(str(params.get("dtype", "")))
            if dtype is None:
                continue

            shape_raw = params.get("shape", [1, 0, 0])
            if not isinstance(shape_raw, list) or len(shape_raw) != 3:
                continue
            if not all(isinstance(item, int) for item in shape_raw):
                continue
            resolution = _shape_label(shape_raw)

            if resolution not in RESOLUTION_ORDER:
                continue

            if dtype not in data[pipeline][device]:
                data[pipeline][device][dtype] = {}

            data[pipeline][device][dtype][resolution] = float(bench["stats"]["median"])

    return data


def _write_latency_fps_table(
    data: dict[PipelineKey, dict[DeviceKey, dict[DTypeKey, dict[str, float]]]],
) -> None:
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\renewcommand{\\arraystretch}{1.2}",
        "\\setlength{\\tabcolsep}{7pt}",
        "\\begin{tabular}{|l|c|r|r|}",
        "\\hline",
        "\\rowcolor[gray]{0.90}",
        "\\textbf{Pipeline} & \\textbf{Resolución} & \\textbf{Latencia (ms)} & \\textbf{Rendimiento (FPS)} \\\\",
        "\\hline",
    ]

    for pipeline in PIPELINE_ORDER:
        for resolution in RESOLUTION_ORDER:
            gpu_s = data[pipeline]["cuda"]["float32"][resolution]
            gpu_fps = 1.0 / gpu_s
            lines.append(
                f"{PIPELINES[pipeline]} & {resolution} & {gpu_s * 1000.0:.3f} & {gpu_fps:.1f} \\\\"
            )
            lines.append("\\hline")

    lines.extend(
        [
            "\\end{tabular}",
            "\\caption{Latencia mediana y rendimiento por resolución}",
            "\\label{tab:resolution_latency_fps_equipo1_float32}",
            "\\end{table}",
            "",
        ]
    )

    (TAB_DIR / "tab_resolution_latency_fps_equipo1_float32.tex").write_text(
        "\n".join(lines)
    )


def _write_scaling_summary_table(
    data: dict[PipelineKey, dict[DeviceKey, dict[DTypeKey, dict[str, float]]]],
) -> None:
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\renewcommand{\\arraystretch}{1.2}",
        "\\setlength{\\tabcolsep}{7pt}",
        "\\begin{tabular}{|l|r|r|r|r|}",
        "\\hline",
        "\\rowcolor[gray]{0.90}",
        "\\textbf{Pipeline} & \\textbf{256->512} & \\textbf{512->1024} & \\textbf{1024->2048} & \\textbf{256->2048} \\\\",
        "\\hline",
    ]

    for pipeline in PIPELINE_ORDER:
        vals = [data[pipeline]["cuda"]["float32"][r] for r in RESOLUTION_ORDER]
        step_1 = vals[1] / vals[0]
        step_2 = vals[2] / vals[1]
        step_3 = vals[3] / vals[2]
        total = vals[3] / vals[0]
        lines.append(
            f"{PIPELINES[pipeline]} & {step_1:.2f}x & {step_2:.2f}x & {step_3:.2f}x & {total:.2f}x \\\\"
        )
        lines.append("\\hline")

    lines.extend(
        [
            "\\end{tabular}",
            "\\caption{Factores de escalado temporal por salto de resolución y factor total acumulado}",
            "\\label{tab:resolution_scaling_summary_equipo1_float32}",
            "\\end{table}",
            "",
        ]
    )

    (TAB_DIR / "tab_resolution_scaling_summary_equipo1_float32.tex").write_text(
        "\n".join(lines)
    )


def _write_mpix_efficiency_table(
    data: dict[PipelineKey, dict[DeviceKey, dict[DTypeKey, dict[str, float]]]],
) -> None:
    latex_nl = "\\\\"
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\renewcommand{\\arraystretch}{1.2}",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\begin{tabular}{|l|c|r|r|r|r|}",
        "\\hline",
        "\\rowcolor[gray]{0.90}",
        "\\textbf{Pipeline} & \\textbf{Resolución} & \\textbf{MPix} & \\textbf{Latencia (ms)} & \\textbf{ms/MPix} & \\textbf{MPix/s} "
        + latex_nl,
        "\\hline",
    ]

    for pipeline in PIPELINE_ORDER:
        for resolution in RESOLUTION_ORDER:
            latency_s = data[pipeline]["cuda"]["float32"][resolution]
            latency_ms = latency_s * 1000.0
            mpix = _resolution_mpix(resolution)
            ms_per_mpix = latency_ms / mpix
            mpix_per_s = mpix / latency_s
            lines.append(
                f"{PIPELINES[pipeline]} & {resolution} & {mpix:.3f} & {latency_ms:.3f} & {ms_per_mpix:.2f} & {mpix_per_s:.0f} \\\\"
            )
            lines.append("\\hline")

    lines.extend(
        [
            "\\end{tabular}",
            "\\caption{Coste temporal normalizado por MPix y rendimiento en MPix/s por resolución}",
            "\\label{tab:resolution_mpix_efficiency_equipo1_float32}",
            "\\end{table}",
            "",
        ]
    )

    (TAB_DIR / "tab_resolution_mpix_efficiency_equipo1_float32.tex").write_text(
        "\n".join(lines)
    )


def _write_scaling_dtype_table(
    data: dict[PipelineKey, dict[DeviceKey, dict[DTypeKey, dict[str, float]]]],
) -> None:
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\renewcommand{\\arraystretch}{1.2}",
        "\\setlength{\\tabcolsep}{7pt}",
        "\\begin{tabular}{|l|c|r|}",
        "\\hline",
        "\\rowcolor[gray]{0.90}",
        "\\textbf{Pipeline} & \\textbf{Dtype} & \\textbf{Factor 256->2048} \\\\",
        "\\hline",
    ]

    for pipeline in PIPELINE_ORDER:
        for dtype in DTYPE_ORDER:
            v256 = data[pipeline]["cuda"][dtype]["256x256"]
            v2048 = data[pipeline]["cuda"][dtype]["2048x2048"]
            factor = v2048 / v256
            lines.append(f"{PIPELINES[pipeline]} & {dtype} & {factor:.2f}x \\\\")
            lines.append("\\hline")

    lines.extend(
        [
            "\\end{tabular}",
            "\\caption{Factor de escalado total en GPU (256->2048) para cada tipo numérico en el Equipo 1.}",
            "\\label{tab:resolution_scaling_by_dtype_equipo1}",
            "\\end{table}",
            "",
        ]
    )

    (TAB_DIR / "tab_resolution_scaling_by_dtype_equipo1.tex").write_text(
        "\n".join(lines)
    )


def _plot_latency(
    data: dict[PipelineKey, dict[DeviceKey, dict[DTypeKey, dict[str, float]]]],
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), sharey=True)
    x = list(range(len(RESOLUTION_ORDER)))
    color = "#2a9d8f"

    for idx, pipeline in enumerate(PIPELINE_ORDER):
        ax = axes[idx]
        y = [
            data[pipeline]["cuda"]["float32"][res] * 1000.0 for res in RESOLUTION_ORDER
        ]
        ax.plot(x, y, marker="o", linewidth=2.2, color=color, label="GPU")

        ax.set_title(PIPELINES[pipeline])
        ax.set_xticks(x)
        ax.set_xticklabels(RESOLUTION_ORDER)
        ax.set_xlabel("Resolución")
        ax.legend(loc="upper left", frameon=True)
        if idx == 0:
            ax.set_ylabel("Latencia mediana (ms)")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(FIG_DIR / "fig_resolution_latency_equipo1_float32.pdf")
    plt.close(fig)


def _plot_scaling(
    data: dict[PipelineKey, dict[DeviceKey, dict[DTypeKey, dict[str, float]]]],
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    step_labels = ["512x512", "1024x1024", "2048x2048"]
    x = list(range(3))
    colors = {"standard_run": "#2a9d8f", "complex_run": "#e76f51"}
    reference = [4.0, 16.0, 64.0]

    for pipeline in PIPELINE_ORDER:
        vals = [data[pipeline]["cuda"]["float32"][r] for r in RESOLUTION_ORDER]
        factors = [vals[1] / vals[0], vals[2] / vals[0], vals[3] / vals[0]]
        ax.plot(
            x,
            factors,
            marker="o",
            linewidth=2.2,
            color=colors[pipeline],
            label=PIPELINES[pipeline],
        )

    ax.plot(
        x,
        reference,
        marker="o",
        linewidth=1.6,
        linestyle="--",
        color="#343a40",
        label="Referencia proporcional",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(step_labels)
    ax.set_xlabel("Resolución respecto a 256x256")
    ax.set_ylabel("Factor de escalado temporal acumulado")
    ax.legend(loc="upper right", frameon=True)

    fig.suptitle(
        "Factor de escalado acumulado por resolución (Equipo 1, float32)", fontsize=12
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(FIG_DIR / "fig_resolution_scaling_factor_equipo1_float32.pdf")
    plt.close(fig)


def _plot_mpix_efficiency(
    data: dict[PipelineKey, dict[DeviceKey, dict[DTypeKey, dict[str, float]]]],
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    x = list(range(len(RESOLUTION_ORDER)))
    colors = {"standard_run": "#2a9d8f", "complex_run": "#e76f51"}

    for pipeline in PIPELINE_ORDER:
        latencies_s = [
            data[pipeline]["cuda"]["float32"][resolution]
            for resolution in RESOLUTION_ORDER
        ]
        mpix_values = [_resolution_mpix(resolution) for resolution in RESOLUTION_ORDER]
        ms_per_mpix = [
            (latency_s * 1000.0) / mpix
            for latency_s, mpix in zip(latencies_s, mpix_values)
        ]
        mpix_per_s = [
            mpix / latency_s for latency_s, mpix in zip(latencies_s, mpix_values)
        ]

        axes[0].plot(
            x,
            ms_per_mpix,
            marker="o",
            linewidth=2.2,
            color=colors[pipeline],
            label=PIPELINES[pipeline],
        )
        axes[1].plot(
            x,
            mpix_per_s,
            marker="o",
            linewidth=2.2,
            color=colors[pipeline],
            label=PIPELINES[pipeline],
        )

    axes[0].set_ylabel("ms/MPix")
    axes[1].set_ylabel("MPix/s")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(RESOLUTION_ORDER)
        ax.set_xlabel("Resolución")
        ax.legend(loc="best", frameon=True)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(FIG_DIR / "fig_resolution_mpix_efficiency_equipo1_float32.pdf")
    plt.close(fig)


def _plot_fps(
    data: dict[PipelineKey, dict[DeviceKey, dict[DTypeKey, dict[str, float]]]],
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), sharey=True)
    x = list(range(len(RESOLUTION_ORDER)))
    color = "#2a9d8f"

    for idx, pipeline in enumerate(PIPELINE_ORDER):
        ax = axes[idx]
        fps = [1.0 / data[pipeline]["cuda"]["float32"][res] for res in RESOLUTION_ORDER]
        ax.plot(x, fps, marker="o", linewidth=2.2, color=color, label="GPU")

        ax.axhline(30.0, color="#d62728", linestyle="--", linewidth=1.4, label="30 FPS")
        ax.axhline(60.0, color="#9467bd", linestyle="--", linewidth=1.4, label="60 FPS")
        ax.set_title(PIPELINES[pipeline])
        ax.set_xticks(x)
        ax.set_xticklabels(RESOLUTION_ORDER)
        ax.set_xlabel("Resolución")
        ax.set_yscale("log")
        ax.legend(loc="upper right", frameon=True)
        if idx == 0:
            ax.set_ylabel("FPS (escala logarítmica)")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(FIG_DIR / "fig_resolution_fps_equipo1_float32.pdf")
    plt.close(fig)


def main() -> None:
    _ensure_dirs()
    data = _load_records()

    _write_latency_fps_table(data)
    _write_scaling_summary_table(data)
    _write_mpix_efficiency_table(data)
    _write_scaling_dtype_table(data)

    _plot_latency(data)
    _plot_scaling(data)
    _plot_mpix_efficiency(data)
    _plot_fps(data)

    print(f"Assets generated in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
