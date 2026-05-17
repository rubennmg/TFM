from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT.parent.parent.parent / "results" / "Equipo1"
OUTPUT_DIR = ROOT / "generated" / "dtype_equipo1"
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
DEVICES: list[DeviceKey] = ["cuda"]
DTYPE_ORDER: list[DTypeKey] = ["float16", "float32", "float64"]
RESOLUTION_ORDER: list[str] = ["256x256", "512x512", "1024x1024", "2048x2048"]
LATEX_NL = "\\\\"


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


def _dtype_label(raw_dtype: str) -> DTypeKey | None:
    if "float16" in raw_dtype:
        return "float16"
    if "float32" in raw_dtype:
        return "float32"
    if "float64" in raw_dtype:
        return "float64"
    return None


def _device_label(device: DeviceKey) -> str:
    return "GPU" if device == "cuda" else "CPU"


def _load_records() -> dict[
    PipelineKey, dict[DeviceKey, dict[DTypeKey, dict[str, float]]]
]:
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


def _write_latency_fps_table_2048(
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
        "\\textbf{Pipeline} & \\textbf{Dtype} & \\textbf{Latencia (ms)} & \\textbf{Rendimiento (FPS)} "
        + LATEX_NL,
        "\\hline",
    ]

    for pipeline in PIPELINE_ORDER:
        for dtype in DTYPE_ORDER:
            latency_s = data[pipeline]["cuda"][dtype]["2048x2048"]
            fps = 1.0 / latency_s
            lines.append(
                f"{PIPELINES[pipeline]} & {dtype} & {latency_s * 1000.0:.3f} & {fps:.1f} {LATEX_NL}"
            )
            lines.append("\\hline")

    lines.extend(
        [
            "\\end{tabular}",
            "\\caption{Impacto del tipo numérico en latencia y rendimiento para \\texttt{2048x2048}}",
            "\\label{tab:dtype_latency_fps_equipo1_2048}",
            "\\end{table}",
            "",
        ]
    )

    (TAB_DIR / "tab_dtype_latency_fps_equipo1_2048.tex").write_text("\n".join(lines))


def _write_ratio_vs_float32_table_2048(
    data: dict[PipelineKey, dict[DeviceKey, dict[DTypeKey, dict[str, float]]]],
) -> None:
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\renewcommand{\\arraystretch}{1.2}",
        "\\setlength{\\tabcolsep}{7pt}",
        "\\begin{tabular}{|l|r|r|}",
        "\\hline",
        "\\rowcolor[gray]{0.90}",
        "\\textbf{Pipeline} & \\textbf{float16 / float32} & \\textbf{float64 / float32} "
        + LATEX_NL,
        "\\hline",
    ]

    for pipeline in PIPELINE_ORDER:
        baseline = data[pipeline]["cuda"]["float32"]["2048x2048"]
        ratio_f16 = data[pipeline]["cuda"]["float16"]["2048x2048"] / baseline
        ratio_f64 = data[pipeline]["cuda"]["float64"]["2048x2048"] / baseline
        lines.append(
            f"{PIPELINES[pipeline]} & {ratio_f16:.2f}x & {ratio_f64:.2f}x {LATEX_NL}"
        )
        lines.append("\\hline")

    lines.extend(
        [
            "\\end{tabular}",
            "\\caption{Relación de latencia respecto a \\texttt{float32} para \\texttt{2048x2048}}",
            "\\label{tab:dtype_ratio_vs_float32_equipo1_2048}",
            "\\end{table}",
            "",
        ]
    )

    (TAB_DIR / "tab_dtype_ratio_vs_float32_equipo1_2048.tex").write_text(
        "\n".join(lines)
    )


def _write_consistency_table_1024_2048(
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
        "\\textbf{Pipeline} & \\textbf{f16/f32 (1024)} & \\textbf{f16/f32 (2048)} & \\textbf{f64/f32 (1024)} & \\textbf{f64/f32 (2048)} "
        + LATEX_NL,
        "\\hline",
    ]

    for pipeline in PIPELINE_ORDER:
        f32_1024 = data[pipeline]["cuda"]["float32"]["1024x1024"]
        f32_2048 = data[pipeline]["cuda"]["float32"]["2048x2048"]
        f16_1024 = data[pipeline]["cuda"]["float16"]["1024x1024"] / f32_1024
        f16_2048 = data[pipeline]["cuda"]["float16"]["2048x2048"] / f32_2048
        f64_1024 = data[pipeline]["cuda"]["float64"]["1024x1024"] / f32_1024
        f64_2048 = data[pipeline]["cuda"]["float64"]["2048x2048"] / f32_2048

        lines.append(
            f"{PIPELINES[pipeline]} & {f16_1024:.2f}x & {f16_2048:.2f}x & {f64_1024:.2f}x & {f64_2048:.2f}x {LATEX_NL}"
        )
        lines.append("\\hline")

    lines.extend(
        [
            "\\end{tabular}",
            "\\caption{Relación de latencia respecto a \\texttt{float32} para \\texttt{1024x1024} y \\texttt{2048x2048}}",
            "\\label{tab:dtype_consistency_1024_2048_equipo1}",
            "\\end{table}",
            "",
        ]
    )

    (TAB_DIR / "tab_dtype_consistency_1024_2048_equipo1.tex").write_text(
        "\n".join(lines)
    )


def _plot_latency_2048(
    data: dict[PipelineKey, dict[DeviceKey, dict[DTypeKey, dict[str, float]]]],
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), sharey=True)
    x = list(range(len(DTYPE_ORDER)))
    color = "#2a9d8f"

    for idx, pipeline in enumerate(PIPELINE_ORDER):
        ax = axes[idx]
        vals = [
            data[pipeline]["cuda"][dtype]["2048x2048"] * 1000.0 for dtype in DTYPE_ORDER
        ]
        ax.bar(x, vals, width=0.5, color=color, label="GPU")

        ax.set_title(PIPELINES[pipeline])
        ax.set_xticks(x)
        ax.set_xticklabels(DTYPE_ORDER)
        ax.set_xlabel("Tipo numérico")
        ax.set_yscale("log")
        ax.legend(loc="upper left", frameon=True)
        if idx == 0:
            ax.set_ylabel("Latencia mediana (ms, escala logarítmica)")

    fig.suptitle(
        "Impacto del tipo numérico en latencia (Equipo 1, 2048x2048)", fontsize=12
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(FIG_DIR / "fig_dtype_latency_equipo1_2048.pdf")
    plt.close(fig)


def _plot_ratio_vs_float32_2048(
    data: dict[PipelineKey, dict[DeviceKey, dict[DTypeKey, dict[str, float]]]],
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), sharey=True)
    x = list(range(len(DTYPE_ORDER)))
    color = "#2a9d8f"

    for idx, pipeline in enumerate(PIPELINE_ORDER):
        ax = axes[idx]
        baseline = data[pipeline]["cuda"]["float32"]["2048x2048"]
        ratios = [
            data[pipeline]["cuda"][dtype]["2048x2048"] / baseline
            for dtype in DTYPE_ORDER
        ]
        ax.plot(x, ratios, marker="o", linewidth=2.2, color=color, label="GPU")

        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
        ax.set_title(PIPELINES[pipeline])
        ax.set_xticks(x)
        ax.set_xticklabels(DTYPE_ORDER)
        ax.set_xlabel("Tipo numérico")
        ax.legend(loc="upper left", frameon=True)
        if idx == 0:
            ax.set_ylabel("Latencia normalizada (float32 = 1)")

    fig.suptitle(
        "Impacto relativo del tipo numérico (Equipo 1, 2048x2048)", fontsize=12
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(FIG_DIR / "fig_dtype_ratio_vs_float32_equipo1_2048.pdf")
    plt.close(fig)


def _plot_ratio_vs_float32_all_resolutions(
    data: dict[PipelineKey, dict[DeviceKey, dict[DTypeKey, dict[str, float]]]],
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    x = list(range(len(RESOLUTION_ORDER)))
    colors = {"float16": "#2a9d8f", "float32": "#457b9d", "float64": "#e76f51"}

    for idx, pipeline in enumerate(PIPELINE_ORDER):
        ax = axes[idx]
        for dtype in DTYPE_ORDER:
            ratios = [
                data[pipeline]["cuda"][dtype][resolution]
                / data[pipeline]["cuda"]["float32"][resolution]
                for resolution in RESOLUTION_ORDER
            ]
            ax.plot(
                x,
                ratios,
                marker="o",
                linewidth=2.2,
                color=colors[dtype],
                label=dtype,
            )

        ax.axhline(1.0, color="#343a40", linestyle="--", linewidth=1.1)
        ax.set_title(PIPELINES[pipeline])
        ax.set_xticks(x)
        ax.set_xticklabels(RESOLUTION_ORDER)
        ax.set_xlabel("Resolución")
        ax.legend(loc="upper left", frameon=True)
        if idx == 0:
            ax.set_ylabel("Latencia relativa (float32 = 1)")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(FIG_DIR / "fig_dtype_ratio_vs_float32_all_resolutions_equipo1.pdf")
    plt.close(fig)


def _plot_fps_all_resolutions(
    data: dict[PipelineKey, dict[DeviceKey, dict[DTypeKey, dict[str, float]]]],
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
    x = list(range(len(RESOLUTION_ORDER)))
    width = 0.24
    offsets = [-width, 0.0, width]
    colors = {"float16": "#2a9d8f", "float32": "#457b9d", "float64": "#e76f51"}

    for idx, pipeline in enumerate(PIPELINE_ORDER):
        ax = axes[idx]
        for dtype, offset in zip(DTYPE_ORDER, offsets):
            fps_vals = [
                1.0 / data[pipeline]["cuda"][dtype][resolution]
                for resolution in RESOLUTION_ORDER
            ]
            ax.bar(
                [value + offset for value in x],
                fps_vals,
                width=width,
                color=colors[dtype],
                label=dtype,
            )

        ax.axhline(30.0, color="#d62728", linestyle="--", linewidth=1.4, label="30 FPS")
        ax.axhline(60.0, color="#9467bd", linestyle="--", linewidth=1.4, label="60 FPS")
        ax.set_title(PIPELINES[pipeline])
        ax.set_xticks(x)
        ax.set_xticklabels(RESOLUTION_ORDER)
        ax.set_xlabel("Resolución")
        ax.set_yscale("log")
        ax.legend(loc="upper right", frameon=True)
        if idx == 0:
            ax.set_ylabel("Rendimiento (FPS, escala logarítmica)")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(FIG_DIR / "fig_dtype_fps_all_resolutions_equipo1.pdf")
    plt.close(fig)


def main() -> None:
    _ensure_dirs()
    data = _load_records()

    _write_latency_fps_table_2048(data)
    _write_ratio_vs_float32_table_2048(data)
    _write_consistency_table_1024_2048(data)

    _plot_latency_2048(data)
    _plot_ratio_vs_float32_2048(data)
    _plot_ratio_vs_float32_all_resolutions(data)
    _plot_fps_all_resolutions(data)

    print(f"Assets generated in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
