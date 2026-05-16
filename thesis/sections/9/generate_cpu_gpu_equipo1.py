from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Sequence, TypedDict, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT.parent.parent.parent / "results" / "Equipo1"
OUTPUT_DIR = ROOT / "generated" / "cpu_gpu_equipo1"
FIG_DIR = OUTPUT_DIR / "figures"
TAB_DIR = OUTPUT_DIR / "tables"

PipelineKey = Literal["standard_run", "complex_run"]
DeviceKey = Literal["cpu", "cuda"]

PIPELINE_ORDER: list[PipelineKey] = ["standard_run", "complex_run"]
PIPELINES: dict[PipelineKey, str] = {
    "standard_run": "standard",
    "complex_run": "complex",
}
RESOLUTION_ORDER: list[str] = ["256x256", "512x512", "1024x1024", "2048x2048"]


class RowRecord(TypedDict):
    pipeline: str
    resolution: str
    cpu_ms: float
    gpu_ms: float
    speedup: float


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


def _load_float32_records() -> dict[PipelineKey, dict[str, dict[DeviceKey, float]]]:
    data: dict[PipelineKey, dict[str, dict[DeviceKey, float]]] = {}

    for pipeline in PIPELINE_ORDER:
        run_path = _run_file(pipeline)
        payload = json.loads(run_path.read_text())
        data[pipeline] = {}

        for bench in payload.get("benchmarks", []):
            params = bench.get("params", {})
            dtype = str(params.get("dtype", ""))
            if "float32" not in dtype:
                continue

            shape_raw = params.get("shape", [1, 0, 0])
            if not isinstance(shape_raw, list) or len(shape_raw) != 3:
                continue
            if not all(isinstance(item, int) for item in shape_raw):
                continue
            resolution = _shape_label(shape_raw)
            device_raw = str(params.get("device", "")).lower()
            if resolution not in RESOLUTION_ORDER or device_raw not in {"cpu", "cuda"}:
                continue
            device: DeviceKey = "cpu" if device_raw == "cpu" else "cuda"

            if resolution not in data[pipeline]:
                data[pipeline][resolution] = {}

            median_s = float(bench["stats"]["median"])
            data[pipeline][resolution][device] = median_s

    return data


def _build_rows(
    data: dict[PipelineKey, dict[str, dict[DeviceKey, float]]],
) -> list[RowRecord]:
    rows: list[RowRecord] = []
    for pipeline in PIPELINE_ORDER:
        for resolution in RESOLUTION_ORDER:
            cpu_s = data[pipeline][resolution]["cpu"]
            gpu_s = data[pipeline][resolution]["cuda"]
            speedup = cpu_s / gpu_s
            rows.append(
                {
                    "pipeline": PIPELINES[pipeline],
                    "resolution": resolution,
                    "cpu_ms": cpu_s * 1000.0,
                    "gpu_ms": gpu_s * 1000.0,
                    "speedup": speedup,
                }
            )
    return rows


def _write_latency_table(rows: list[RowRecord]) -> None:
    latex_nl = "\\\\"
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\renewcommand{\\arraystretch}{1.2}",
        "\\setlength{\\tabcolsep}{8pt}",
        "\\begin{tabular}{|l|c|r|r|r|}",
        "\\hline",
        "\\rowcolor[gray]{0.90}",
        "\\textbf{Pipeline} & \\textbf{Resolución} & \\textbf{CPU (ms)} & \\textbf{GPU (ms)} & \\textbf{Speedup} "
        + latex_nl,
        "\\hline",
    ]

    for row in rows:
        lines.append(
            f"{row['pipeline']} & {row['resolution']} & {row['cpu_ms']:.3f} & {row['gpu_ms']:.3f} & {row['speedup']:.2f}x {latex_nl}"
        )
        lines.append("\\hline")

    lines.extend(
        [
            "\\end{tabular}",
            "\\caption{Comparativa CPU vs GPU}",
            "\\label{tab:cpu_gpu_equipo1_float32_latency}",
            "\\end{table}",
            "",
        ]
    )

    (TAB_DIR / "tab_cpu_gpu_float32_latency.tex").write_text("\n".join(lines))


def _write_summary_table(rows: list[RowRecord]) -> None:
    grouped: dict[str, list[float]] = {name: [] for name in PIPELINES.values()}
    by_pipeline_res: dict[tuple[str, str], float] = {}

    for row in rows:
        pipeline_name = str(row["pipeline"])
        resolution = str(row["resolution"])
        speedup = float(row["speedup"])
        grouped[pipeline_name].append(speedup)
        by_pipeline_res[(pipeline_name, resolution)] = speedup

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\renewcommand{\\arraystretch}{1.2}",
        "\\setlength{\\tabcolsep}{8pt}",
        "\\begin{tabular}{|l|r|r|r|r|}",
        "\\hline",
        "\\rowcolor[gray]{0.90}",
        "\\textbf{Pipeline} & \\textbf{Speedup mínimo} & \\textbf{Speedup mediano} & \\textbf{Speedup medio} & \\textbf{Speedup en 2048x2048} \\\\",
        "\\hline",
    ]

    for pipeline_name, values in grouped.items():
        values_sorted = sorted(values)
        median_value = (values_sorted[1] + values_sorted[2]) / 2.0
        mean_value = sum(values_sorted) / len(values_sorted)
        speedup_2048 = by_pipeline_res[(pipeline_name, "2048x2048")]

        lines.append(
            f"{pipeline_name} & {min(values_sorted):.2f}x & {median_value:.2f}x & {mean_value:.2f}x & {speedup_2048:.2f}x \\\\"
        )
        lines.append("\\hline")

    lines.extend(
        [
            "\\end{tabular}",
            "\\caption{Resumen de speedup CPU/GPU por pipeline en el Equipo 1 (\\texttt{float32}).}",
            "\\label{tab:cpu_gpu_equipo1_speedup_summary}",
            "\\end{table}",
            "",
        ]
    )

    (TAB_DIR / "tab_cpu_gpu_speedup_summary.tex").write_text("\n".join(lines))


def _plot_latency(data: dict[PipelineKey, dict[str, dict[DeviceKey, float]]]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), sharey=True)

    colors = {"cpu": "#1f4e79", "cuda": "#2a9d8f"}
    x = list(range(len(RESOLUTION_ORDER)))

    for idx, (pipeline, title) in enumerate(PIPELINES.items()):
        cpu_ms = [data[pipeline][r]["cpu"] * 1000.0 for r in RESOLUTION_ORDER]
        gpu_ms = [data[pipeline][r]["cuda"] * 1000.0 for r in RESOLUTION_ORDER]
        ax = axes[idx]
        ax.plot(x, cpu_ms, marker="o", linewidth=2.2, color=colors["cpu"], label="CPU")
        ax.plot(x, gpu_ms, marker="o", linewidth=2.2, color=colors["cuda"], label="GPU")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(RESOLUTION_ORDER)
        ax.set_xlabel("Resolución")
        ax.set_yscale("log")
        ax.legend(loc="upper left", frameon=True)
        if idx == 0:
            ax.set_ylabel("Latencia mediana (ms, escala logarítmica)")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    fig.savefig(FIG_DIR / "fig_cpu_gpu_latency_equipo1_float32.pdf")
    plt.close(fig)


def _plot_speedup(rows: list[RowRecord]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.0, 4.8))

    width = 0.36
    x = list(range(len(RESOLUTION_ORDER)))

    standard = [float(r["speedup"]) for r in rows if r["pipeline"] == "standard"]
    complex_ = [float(r["speedup"]) for r in rows if r["pipeline"] == "complex"]

    ax.bar(
        [v - width / 2 for v in x],
        standard,
        width=width,
        color="#457b9d",
        label="standard",
    )
    ax.bar(
        [v + width / 2 for v in x],
        complex_,
        width=width,
        color="#e76f51",
        label="complex",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(RESOLUTION_ORDER)
    ax.set_xlabel("Resolución")
    ax.set_ylabel("Speedup CPU/GPU")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.legend(loc="upper left")

    for i, value in enumerate(standard):
        ax.text(
            i - width / 2,
            value + 1.0,
            f"{value:.1f}x",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for i, value in enumerate(complex_):
        ax.text(
            i + width / 2,
            value + 1.0,
            f"{value:.1f}x",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_cpu_gpu_speedup_equipo1_float32.pdf")
    plt.close(fig)


def _plot_fps(data: dict[PipelineKey, dict[str, dict[DeviceKey, float]]]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), sharey=True)

    colors = {"cpu": "#1f4e79", "cuda": "#2a9d8f"}
    labels = {"cpu": "CPU", "cuda": "GPU"}
    x = list(range(len(RESOLUTION_ORDER)))

    for idx, (pipeline, title) in enumerate(PIPELINES.items()):
        ax = axes[idx]
        for device in ["cpu", "cuda"]:
            dev = cast(DeviceKey, device)
            fps_values = [
                1.0 / data[pipeline][resolution][dev] for resolution in RESOLUTION_ORDER
            ]
            ax.plot(
                x,
                fps_values,
                marker="o",
                linewidth=2.2,
                color=colors[device],
                label=labels[device],
            )

        ax.axhline(30.0, color="#6c757d", linestyle="--", linewidth=1.1)
        ax.axhline(60.0, color="#343a40", linestyle="--", linewidth=1.1)
        ax.text(
            0.02,
            30.0,
            "30 FPS",
            color="#6c757d",
            fontsize=8,
            va="bottom",
            ha="left",
            transform=ax.get_yaxis_transform(),
        )
        ax.text(
            0.02,
            60.0,
            "60 FPS",
            color="#343a40",
            fontsize=8,
            va="bottom",
            ha="left",
            transform=ax.get_yaxis_transform(),
        )

        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(RESOLUTION_ORDER)
        ax.set_xlabel("Resolución")
        ax.set_yscale("log")
        ax.legend(loc="upper right", frameon=True)
        if idx == 0:
            ax.set_ylabel("FPS (escala logarítmica)")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    fig.savefig(FIG_DIR / "fig_cpu_gpu_fps_equipo1_float32.pdf")
    plt.close(fig)


def _write_realtime_table(rows: list[RowRecord]) -> None:
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\renewcommand{\\arraystretch}{1.2}",
        "\\setlength{\\tabcolsep}{8pt}",
        "\\begin{tabular}{|l|c|c|r|r|}",
        "\\hline",
        "\\rowcolor[gray]{0.90}",
        "\\textbf{Pipeline} & \\textbf{Resolución} & \\textbf{Dispositivo} & \\textbf{FPS} & \\textbf{Cumple 30 FPS} \\\\",
        "\\hline",
    ]

    for row in rows:
        pipeline = row["pipeline"]
        resolution = row["resolution"]
        cpu_ms = row["cpu_ms"]
        gpu_ms = row["gpu_ms"]
        cpu_fps = 1000.0 / cpu_ms
        gpu_fps = 1000.0 / gpu_ms

        lines.append(
            f"{pipeline} & {resolution} & CPU & {cpu_fps:.1f} & {'Sí' if cpu_fps >= 30.0 else 'No'} \\\\"
        )
        lines.append("\\hline")
        lines.append(
            f"{pipeline} & {resolution} & GPU & {gpu_fps:.1f} & {'Sí' if gpu_fps >= 30.0 else 'No'} \\\\"
        )
        lines.append("\\hline")

    lines.extend(
        [
            "\\end{tabular}",
            "\\caption{Viabilidad temporal a 30 FPS para la comparativa CPU vs GPU en el Equipo 1 (\\texttt{float32}).}",
            "\\label{tab:cpu_gpu_equipo1_realtime_30fps}",
            "\\end{table}",
            "",
        ]
    )

    (TAB_DIR / "tab_cpu_gpu_realtime_30fps.tex").write_text("\n".join(lines))


def main() -> None:
    _ensure_dirs()
    data = _load_float32_records()
    rows = _build_rows(data)

    _write_latency_table(rows)
    _write_summary_table(rows)
    _write_realtime_table(rows)
    _plot_latency(data)
    _plot_speedup(rows)
    _plot_fps(data)

    print(f"Assets generated in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
