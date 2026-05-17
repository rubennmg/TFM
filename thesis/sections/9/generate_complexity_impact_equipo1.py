from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT.parent.parent.parent / "results" / "Equipo1"
OUTPUT_DIR = ROOT / "generated" / "complexity_equipo1"
FIG_DIR = OUTPUT_DIR / "figures"
TAB_DIR = OUTPUT_DIR / "tables"

PipelineKey = Literal["standard_run", "complex_run"]
DeviceKey = Literal["cpu", "cuda"]

PIPELINES: dict[PipelineKey, str] = {
    "standard_run": "standard",
    "complex_run": "complex",
}
PIPELINE_ORDER: list[PipelineKey] = ["standard_run", "complex_run"]
DEVICES: list[DeviceKey] = ["cuda"]
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


def _resolution_mpix(resolution: str) -> float:
    width, height = (int(value) for value in resolution.split("x"))
    return (width * height) / 1_000_000.0


def _load_float32() -> dict[PipelineKey, dict[DeviceKey, dict[str, float]]]:
    data: dict[PipelineKey, dict[DeviceKey, dict[str, float]]] = {
        "standard_run": {"cpu": {}, "cuda": {}},
        "complex_run": {"cpu": {}, "cuda": {}},
    }

    for pipeline in PIPELINE_ORDER:
        payload = json.loads(_run_file(pipeline).read_text())
        for bench in payload.get("benchmarks", []):
            params = bench.get("params", {})
            dtype_raw = str(params.get("dtype", ""))
            if "float32" not in dtype_raw:
                continue

            device_raw = str(params.get("device", "")).lower()
            if device_raw not in {"cpu", "cuda"}:
                continue
            device: DeviceKey = "cpu" if device_raw == "cpu" else "cuda"

            shape_raw = params.get("shape", [1, 0, 0])
            if not isinstance(shape_raw, list) or len(shape_raw) != 3:
                continue
            if not all(isinstance(item, int) for item in shape_raw):
                continue
            resolution = _shape_label(shape_raw)
            if resolution not in RESOLUTION_ORDER:
                continue

            data[pipeline][device][resolution] = float(bench["stats"]["median"])

    return data


def _write_ratio_table(
    data: dict[PipelineKey, dict[DeviceKey, dict[str, float]]],
) -> None:
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\renewcommand{\\arraystretch}{1.2}",
        "\\setlength{\\tabcolsep}{8pt}",
        "\\begin{tabular}{|c|r|}",
        "\\hline",
        "\\rowcolor[gray]{0.90}",
        "\\textbf{Resolución} & \\textbf{Ratio GPU (complex/standard)} " + LATEX_NL,
        "\\hline",
    ]

    for resolution in RESOLUTION_ORDER:
        ratio_gpu = (
            data["complex_run"]["cuda"][resolution]
            / data["standard_run"]["cuda"][resolution]
        )
        lines.append(f"{resolution} & {ratio_gpu:.2f}x {LATEX_NL}")
        lines.append("\\hline")

    lines.extend(
        [
            "\\end{tabular}",
            "\\caption{Impacto relativo de la complejidad del \\textit{pipeline}}",
            "\\label{tab:complexity_ratio_equipo1_float32}",
            "\\end{table}",
            "",
        ]
    )

    (TAB_DIR / "tab_complexity_ratio_equipo1_float32.tex").write_text("\n".join(lines))


def _write_delta_fps_table(
    data: dict[PipelineKey, dict[DeviceKey, dict[str, float]]],
) -> None:
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\renewcommand{\\arraystretch}{1.2}",
        "\\setlength{\\tabcolsep}{6pt}",
        "\\begin{tabular}{|c|r|r|r|r|r|}",
        "\\hline",
        "\\rowcolor[gray]{0.90}",
        "\\textbf{Resolución} & \\textbf{Standard (ms)} & \\textbf{Complex (ms)} & \\textbf{Delta (ms)} & \\textbf{Caída FPS (\\%)} "
        + LATEX_NL,
        "\\hline",
    ]

    for resolution in RESOLUTION_ORDER:
        standard_s = data["standard_run"]["cuda"][resolution]
        complex_s = data["complex_run"]["cuda"][resolution]
        delta_ms = (complex_s - standard_s) * 1000.0
        fps_standard = 1.0 / standard_s
        fps_complex = 1.0 / complex_s
        fps_drop = (1.0 - (fps_complex / fps_standard)) * 100.0

        lines.append(
            f"{resolution} & {standard_s * 1000.0:.3f} & {complex_s * 1000.0:.3f} & {delta_ms:.3f} & {fps_drop:.1f} {LATEX_NL}"
        )
        lines.append("\\hline")

    lines.extend(
        [
            "\\end{tabular}",
            "\\caption{Impacto absoluto de complejidad en latencia y caída de FPS}",
            "\\label{tab:complexity_delta_latency_fps_equipo1_float32}",
            "\\end{table}",
            "",
        ]
    )

    (TAB_DIR / "tab_complexity_delta_latency_fps_equipo1_float32.tex").write_text(
        "\n".join(lines)
    )


def _write_delta_mpix_table(
    data: dict[PipelineKey, dict[DeviceKey, dict[str, float]]],
) -> None:
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\renewcommand{\\arraystretch}{1.2}",
        "\\setlength{\\tabcolsep}{8pt}",
        "\\begin{tabular}{|c|r|r|r|}",
        "\\hline",
        "\\rowcolor[gray]{0.90}",
        "\\textbf{Resolución} & \\textbf{MPix} & \\textbf{Delta (ms)} & \\textbf{Delta (ms/MPix)} "
        + LATEX_NL,
        "\\hline",
    ]

    for resolution in RESOLUTION_ORDER:
        standard_s = data["standard_run"]["cuda"][resolution]
        complex_s = data["complex_run"]["cuda"][resolution]
        mpix = _resolution_mpix(resolution)
        delta_ms = (complex_s - standard_s) * 1000.0
        delta_ms_per_mpix = delta_ms / mpix

        lines.append(
            f"{resolution} & {mpix:.3f} & {delta_ms:.3f} & {delta_ms_per_mpix:.2f} {LATEX_NL}"
        )
        lines.append("\\hline")

    lines.extend(
        [
            "\\end{tabular}",
            "\\caption{Incremento absoluto de latencia normalizado por megapíxel}",
            "\\label{tab:complexity_delta_mpix_equipo1_float32}",
            "\\end{table}",
            "",
        ]
    )

    (TAB_DIR / "tab_complexity_delta_mpix_equipo1_float32.tex").write_text(
        "\n".join(lines)
    )


def _write_summary_table(
    data: dict[PipelineKey, dict[DeviceKey, dict[str, float]]],
) -> None:
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\renewcommand{\\arraystretch}{1.2}",
        "\\setlength{\\tabcolsep}{8pt}",
        "\\begin{tabular}{|r|r|r|}",
        "\\hline",
        "\\rowcolor[gray]{0.90}",
        "\\textbf{Ratio mínimo} & \\textbf{Ratio mediano} & \\textbf{Ratio medio} "
        + LATEX_NL,
        "\\hline",
    ]

    ratios = [
        data["complex_run"]["cuda"][resolution]
        / data["standard_run"]["cuda"][resolution]
        for resolution in RESOLUTION_ORDER
    ]
    sorted_ratios = sorted(ratios)
    median_ratio = (sorted_ratios[1] + sorted_ratios[2]) / 2.0
    mean_ratio = sum(ratios) / len(ratios)
    lines.append(
        f"{min(ratios):.2f}x & {median_ratio:.2f}x & {mean_ratio:.2f}x {LATEX_NL}"
    )
    lines.append("\\hline")

    lines.extend(
        [
            "\\end{tabular}",
            "\\caption{Resumen agregado del sobrecoste de complejidad en \\texttt{float32} (Equipo 1).}",
            "\\label{tab:complexity_summary_equipo1_float32}",
            "\\end{table}",
            "",
        ]
    )

    (TAB_DIR / "tab_complexity_summary_equipo1_float32.tex").write_text(
        "\n".join(lines)
    )


def _plot_latency(data: dict[PipelineKey, dict[DeviceKey, dict[str, float]]]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.2, 4.8))

    x = list(range(len(RESOLUTION_ORDER)))
    colors = {"standard_run": "#2a9d8f", "complex_run": "#e76f51"}

    for pipeline in PIPELINE_ORDER:
        latency_ms = [
            data[pipeline]["cuda"][resolution] * 1000.0
            for resolution in RESOLUTION_ORDER
        ]
        ax.plot(
            x,
            latency_ms,
            marker="o",
            linewidth=2.2,
            color=colors[pipeline],
            label=PIPELINES[pipeline],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(RESOLUTION_ORDER)
    ax.set_xlabel("Resolución")
    ax.set_ylabel("Latencia mediana (ms)")
    ax.legend(loc="upper left", frameon=True)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_complexity_latency_equipo1_float32.pdf")
    plt.close(fig)


def _plot_ratio(data: dict[PipelineKey, dict[DeviceKey, dict[str, float]]]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.2, 4.8))

    x = list(range(len(RESOLUTION_ORDER)))
    ratio_gpu = [
        data["complex_run"]["cuda"][resolution]
        / data["standard_run"]["cuda"][resolution]
        for resolution in RESOLUTION_ORDER
    ]

    ax.plot(x, ratio_gpu, marker="o", linewidth=2.2, color="#2a9d8f", label="GPU")

    ax.set_xticks(x)
    ax.set_xticklabels(RESOLUTION_ORDER)
    ax.set_xlabel("Resolución")
    ax.set_ylabel("Ratio complex/standard")
    ax.set_title("Sobrecoste relativo por complejidad del pipeline (Equipo 1, float32)")
    ax.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_complexity_ratio_equipo1_float32.pdf")
    plt.close(fig)


def _plot_delta_latency(
    data: dict[PipelineKey, dict[DeviceKey, dict[str, float]]],
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.2, 4.8))

    x = list(range(len(RESOLUTION_ORDER)))
    delta_gpu = [
        (
            data["complex_run"]["cuda"][resolution]
            - data["standard_run"]["cuda"][resolution]
        )
        * 1000.0
        for resolution in RESOLUTION_ORDER
    ]

    ax.bar(x, delta_gpu, width=0.5, color="#2a9d8f", label="GPU")

    ax.set_xticks(x)
    ax.set_xticklabels(RESOLUTION_ORDER)
    ax.set_xlabel("Resolución")
    ax.set_ylabel("Incremento de latencia (ms)")
    ax.legend(loc="upper left", frameon=True)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_complexity_delta_latency_equipo1_float32.pdf")
    plt.close(fig)


def _plot_fps_drop(data: dict[PipelineKey, dict[DeviceKey, dict[str, float]]]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.2, 4.8))

    x = list(range(len(RESOLUTION_ORDER)))
    fps_drop_gpu = []

    for resolution in RESOLUTION_ORDER:
        fps_std_gpu = 1.0 / data["standard_run"]["cuda"][resolution]
        fps_cpx_gpu = 1.0 / data["complex_run"]["cuda"][resolution]

        fps_drop_gpu.append((1.0 - (fps_cpx_gpu / fps_std_gpu)) * 100.0)

    ax.plot(x, fps_drop_gpu, marker="o", linewidth=2.2, color="#2a9d8f", label="GPU")

    ax.set_xticks(x)
    ax.set_xticklabels(RESOLUTION_ORDER)
    ax.set_xlabel("Resolución")
    ax.set_ylabel("Caída de FPS (%)")
    ax.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_complexity_fps_drop_equipo1_float32.pdf")
    plt.close(fig)


def main() -> None:
    _ensure_dirs()
    data = _load_float32()

    _write_ratio_table(data)
    _write_delta_fps_table(data)
    _write_delta_mpix_table(data)
    _write_summary_table(data)

    _plot_latency(data)
    _plot_ratio(data)
    _plot_delta_latency(data)
    _plot_fps_drop(data)

    print(f"Assets generated in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
