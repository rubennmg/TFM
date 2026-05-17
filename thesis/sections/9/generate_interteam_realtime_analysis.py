from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Sequence, TypeAlias, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT.parent.parent.parent / "results"
OUTPUT_DIR = ROOT / "generated" / "interteam"
FIG_DIR = OUTPUT_DIR / "figures"
TAB_DIR = OUTPUT_DIR / "tables"

TeamKey: TypeAlias = Literal["Equipo1", "Equipo2", "Equipo3"]
PipelineKey: TypeAlias = Literal["standard_run", "complex_run"]
DeviceKey: TypeAlias = Literal["cpu", "cuda"]
InterteamData: TypeAlias = dict[TeamKey, dict[PipelineKey, dict[DeviceKey, dict[str, float]]]]

EQUIPO1: TeamKey = "Equipo1"
EQUIPO2: TeamKey = "Equipo2"
EQUIPO3: TeamKey = "Equipo3"
STANDARD_RUN: PipelineKey = "standard_run"
COMPLEX_RUN: PipelineKey = "complex_run"
CPU: DeviceKey = "cpu"
CUDA: DeviceKey = "cuda"

TEAM_ORDER: tuple[TeamKey, ...] = (EQUIPO1, EQUIPO2, EQUIPO3)
EMBEDDED_TEAM_ORDER: tuple[TeamKey, ...] = (EQUIPO2, EQUIPO3)
PIPELINE_ORDER: tuple[PipelineKey, ...] = (STANDARD_RUN, COMPLEX_RUN)
PIPELINE_LABELS: dict[PipelineKey, str] = {
    STANDARD_RUN: "standard",
    COMPLEX_RUN: "complex",
}
DEVICE_ORDER: tuple[DeviceKey, ...] = (CUDA,)
DEVICE_LABELS: dict[DeviceKey, str] = {CPU: "CPU", CUDA: "GPU"}
RESOLUTION_ORDER: list[str] = ["256x256", "512x512", "1024x1024", "2048x2048"]

RUN_FILES: dict[TeamKey, dict[PipelineKey, str]] = {
    EQUIPO1: {
        STANDARD_RUN: "0006_standard_run.json",
        COMPLEX_RUN: "0001_complex_run.json",
    },
    EQUIPO2: {
        STANDARD_RUN: "0008_standard_run.json",
        COMPLEX_RUN: "0003_complex_run.json",
    },
    EQUIPO3: {
        STANDARD_RUN: "0006_standard_run.json",
        COMPLEX_RUN: "0001_complex_run.json",
    },
}

TEAM_COLORS: dict[TeamKey, str] = {
    EQUIPO1: "#1f4e79",
    EQUIPO2: "#2a9d8f",
    EQUIPO3: "#e76f51",
}

LATEX_NL = "\\\\"


def _ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)


def _shape_label(shape: Sequence[int]) -> str:
    return f"{shape[1]}x{shape[2]}"


def _load_float32() -> InterteamData:
    data: InterteamData = {
        team: {pipeline: {CPU: {}, CUDA: {}} for pipeline in PIPELINE_ORDER}
        for team in TEAM_ORDER
    }

    for team in TEAM_ORDER:
        for pipeline in PIPELINE_ORDER:
            file_path = RESULTS_DIR / team / RUN_FILES[team][pipeline]
            payload = cast(dict[str, Any], json.loads(file_path.read_text()))
            benchmarks = payload.get("benchmarks", [])
            if not isinstance(benchmarks, list):
                continue

            for bench in benchmarks:
                if not isinstance(bench, dict):
                    continue

                params = bench.get("params", {})
                if not isinstance(params, dict):
                    continue

                if "float32" not in str(params.get("dtype", "")):
                    continue

                device_raw = str(params.get("device", "")).lower()
                if device_raw not in {"cpu", "cuda"}:
                    continue
                device: DeviceKey = CPU if device_raw == CPU else CUDA

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

                data[team][pipeline][device][resolution] = float(median)

    return data


def _write_latency_fps_table(
    data: InterteamData,
) -> None:
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\renewcommand{\\arraystretch}{1.15}",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\begin{tabular}{|c|c|c|r|r|}",
        "\\hline",
        "\\rowcolor[gray]{0.90}",
        "\\textbf{Equipo} & \\textbf{Pipeline} & \\textbf{Resolución} & \\textbf{Latencia (ms)} & \\textbf{Rendimiento (FPS)} "
        + LATEX_NL,
        "\\hline",
    ]

    for team in TEAM_ORDER:
        for pipeline in PIPELINE_ORDER:
            for resolution in RESOLUTION_ORDER:
                latency_s = data[team][pipeline][CUDA][resolution]
                fps = 1.0 / latency_s
                lines.append(
                    f"{team} & {PIPELINE_LABELS[pipeline]} & {resolution} & {latency_s * 1000.0:.3f} & {fps:.1f} {LATEX_NL}"
                )
                lines.append("\\hline")

    lines.extend(
        [
            "\\end{tabular}",
            "\\caption{Latencia y rendimiento por plataforma}",
            "\\label{tab:interteam_latency_fps_float32}",
            "\\end{table}",
            "",
        ]
    )

    (TAB_DIR / "tab_interteam_latency_fps_float32.tex").write_text("\n".join(lines))


def _write_realtime_thresholds_table(
    data: InterteamData,
) -> None:
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\renewcommand{\\arraystretch}{1.2}",
        "\\setlength{\\tabcolsep}{8pt}",
        "\\begin{tabular}{|c|c|c|}",
        "\\hline",
        "\\rowcolor[gray]{0.90}",
        "\\textbf{Equipo} & \\textbf{Pipeline} & \\textbf{Máxima resolución GPU con $\\geq 30$ FPS} "
        + LATEX_NL,
        "\\hline",
    ]

    for team in TEAM_ORDER:
        for pipeline in PIPELINE_ORDER:
            max_ok = "Ninguna"
            for resolution in RESOLUTION_ORDER:
                fps = 1.0 / data[team][pipeline][CUDA][resolution]
                if fps >= 30.0:
                    max_ok = resolution
            lines.append(f"{team} & {PIPELINE_LABELS[pipeline]} & {max_ok} {LATEX_NL}")
            lines.append("\\hline")

    lines.extend(
        [
            "\\end{tabular}",
            "\\caption{Resolución máxima que mantiene al menos 30 FPS en GPU para \\texttt{float32}.}",
            "\\label{tab:interteam_realtime_thresholds_30fps}",
            "\\end{table}",
            "",
        ]
    )

    (TAB_DIR / "tab_interteam_realtime_thresholds_30fps.tex").write_text(
        "\n".join(lines)
    )


def _write_normalized_table(
    data: InterteamData,
) -> None:
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\renewcommand{\\arraystretch}{1.15}",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\begin{tabular}{|c|c|r|r|r|r|}",
        "\\hline",
        "\\rowcolor[gray]{0.90}",
        "\\textbf{Equipo} & \\textbf{Pipeline} & \\textbf{256x256} & \\textbf{512x512} & \\textbf{1024x1024} & \\textbf{2048x2048} "
        + LATEX_NL,
        "\\hline",
    ]

    for team in TEAM_ORDER:
        for pipeline in PIPELINE_ORDER:
            ratios = [
                data[team][pipeline][CUDA][resolution]
                / data[EQUIPO1][pipeline][CUDA][resolution]
                for resolution in RESOLUTION_ORDER
            ]
            lines.append(
                f"{team} & {PIPELINE_LABELS[pipeline]} & "
                + " & ".join(f"{ratio:.2f}x" for ratio in ratios)
                + f" {LATEX_NL}"
            )
            lines.append("\\hline")

    lines.extend(
        [
            "\\end{tabular}",
            "\\caption{Latencia GPU normalizada respecto al Equipo 1}",
            "\\label{tab:interteam_normalized_vs_equipo1_gpu}",
            "\\end{table}",
            "",
        ]
    )

    (TAB_DIR / "tab_interteam_normalized_vs_equipo1_gpu.tex").write_text(
        "\n".join(lines)
    )


def _write_self_scaling_table(
    data: InterteamData,
) -> None:
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\renewcommand{\\arraystretch}{1.15}",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\begin{tabular}{|c|c|r|r|r|r|}",
        "\\hline",
        "\\rowcolor[gray]{0.90}",
        "\\textbf{Equipo} & \\textbf{Pipeline} & \\textbf{256x256} & \\textbf{512x512} & \\textbf{1024x1024} & \\textbf{2048x2048} "
        + LATEX_NL,
        "\\hline",
    ]

    for team in TEAM_ORDER:
        for pipeline in PIPELINE_ORDER:
            base = data[team][pipeline][CUDA]["256x256"]
            ratios = [
                data[team][pipeline][CUDA][resolution] / base
                for resolution in RESOLUTION_ORDER
            ]
            lines.append(
                f"{team} & {PIPELINE_LABELS[pipeline]} & "
                + " & ".join(f"{ratio:.2f}x" for ratio in ratios)
                + f" {LATEX_NL}"
            )
            lines.append("\\hline")

    lines.extend(
        [
            "\\end{tabular}",
            "\\caption{Escalado interno de latencia respecto a \\texttt{256x256}}",
            "\\label{tab:interteam_self_scaling_256_gpu}",
            "\\end{table}",
            "",
        ]
    )

    (TAB_DIR / "tab_interteam_self_scaling_256_gpu.tex").write_text("\n".join(lines))


def _plot_latency(
    data: InterteamData,
    pipeline: PipelineKey,
    output_name: str,
    *,
    log_scale: bool,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    x = list(range(len(RESOLUTION_ORDER)))
    for team in TEAM_ORDER:
        y = [
            data[team][pipeline][CUDA][resolution] * 1000.0
            for resolution in RESOLUTION_ORDER
        ]
        ax.plot(x, y, marker="o", linewidth=2.2, color=TEAM_COLORS[team], label=team)

    ax.set_xticks(x)
    ax.set_xticklabels(RESOLUTION_ORDER)
    ax.set_xlabel("Resolución")
    if log_scale:
        ax.set_yscale("log")
        ax.set_ylabel("Latencia mediana (ms, escala logarítmica)")
    else:
        ax.set_ylabel("Latencia mediana (ms)")
    ax.legend(loc="upper left", frameon=True)

    fig.tight_layout()
    fig.savefig(FIG_DIR / output_name)
    plt.close(fig)


def _plot_realtime_map(
    data: InterteamData,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharey=True)
    im: Any | None = None

    for idx, pipeline in enumerate(PIPELINE_ORDER):
        ax = axes[idx]

        labels: list[str] = []
        matrix: list[list[float]] = []
        for team in EMBEDDED_TEAM_ORDER:
            row: list[float] = []
            labels.append(team)
            for resolution in RESOLUTION_ORDER:
                fps = 1.0 / data[team][pipeline][CUDA][resolution]
                row.append(1.0 if fps >= 30.0 else 0.0)
            matrix.append(row)

        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)
        ax.set_title(PIPELINE_LABELS[pipeline])
        ax.set_xticks(list(range(len(RESOLUTION_ORDER))))
        ax.set_xticklabels(RESOLUTION_ORDER)
        ax.set_yticks(list(range(len(labels))))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Resolución")
        if idx == 0:
            ax.set_ylabel("Equipo")

        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                ax.text(
                    j,
                    i,
                    "Sí" if value >= 0.5 else "No",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    if im is not None:
        cbar = fig.colorbar(im, ax=axes, shrink=0.92)
        cbar.set_label("Cumplimiento de 30 FPS")
    fig.suptitle(
        "Mapa de viabilidad GPU en tiempo real (30 FPS) para float32", fontsize=12
    )
    fig.subplots_adjust(top=0.84, right=0.92, wspace=0.35)
    fig.savefig(FIG_DIR / "fig_interteam_fps_realtime_map_float32.pdf")
    plt.close(fig)


def _plot_fps_comparison(
    data: InterteamData,
    output_name: str,
    *,
    log_scale: bool,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharey=True)
    x = list(range(len(RESOLUTION_ORDER)))
    for idx, pipeline in enumerate(PIPELINE_ORDER):
        ax = axes[idx]
        for team in TEAM_ORDER:
            fps_values = [
                1.0 / data[team][pipeline][CUDA][resolution]
                for resolution in RESOLUTION_ORDER
            ]
            ax.plot(
                x,
                fps_values,
                marker="o",
                linewidth=2.2,
                color=TEAM_COLORS[team],
                label=team,
            )

        ax.set_title(PIPELINE_LABELS[pipeline])
        ax.set_xticks(x)
        ax.set_xticklabels(RESOLUTION_ORDER)
        ax.set_xlabel("Resolución")
        if log_scale:
            ax.set_yscale("log")
        if idx == 0:
            ylabel = "FPS GPU (escala logarítmica)" if log_scale else "FPS GPU"
            ax.set_ylabel(ylabel)

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
        ax.legend(loc="upper right", frameon=True)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(FIG_DIR / output_name)
    plt.close(fig)


def _plot_embedded_fps_comparison(
    data: InterteamData,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharey=True)
    x = list(range(len(RESOLUTION_ORDER)))
    for idx, pipeline in enumerate(PIPELINE_ORDER):
        ax = axes[idx]
        for team in EMBEDDED_TEAM_ORDER:
            fps_values = [
                1.0 / data[team][pipeline][CUDA][resolution]
                for resolution in RESOLUTION_ORDER
            ]
            ax.plot(
                x,
                fps_values,
                marker="o",
                linewidth=2.2,
                color=TEAM_COLORS[team],
                label=team,
            )

        ax.set_title(PIPELINE_LABELS[pipeline])
        ax.set_xticks(x)
        ax.set_xticklabels(RESOLUTION_ORDER)
        ax.set_xlabel("Resolución")
        ax.set_yscale("log")
        if idx == 0:
            ax.set_ylabel("FPS GPU (escala logarítmica)")

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
        ax.legend(loc="upper right", frameon=True)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(FIG_DIR / "fig_interteam_embedded_fps_comparison_log_float32.pdf")
    plt.close(fig)


def _plot_gpu_scaling(
    data: InterteamData,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharey=True)
    x = list(range(len(RESOLUTION_ORDER)))
    for idx, pipeline in enumerate(PIPELINE_ORDER):
        ax = axes[idx]
        for team in TEAM_ORDER:
            base = data[team][pipeline][CUDA]["256x256"]
            normalized = [
                data[team][pipeline][CUDA][resolution] / base
                for resolution in RESOLUTION_ORDER
            ]
            ax.plot(
                x, normalized, marker="o", linewidth=2.2, color=TEAM_COLORS[team], label=team
            )

        ax.set_title(PIPELINE_LABELS[pipeline])
        ax.set_xticks(x)
        ax.set_xticklabels(RESOLUTION_ORDER)
        ax.set_xlabel("Resolución")
        if idx == 0:
            ax.set_ylabel("Latencia normalizada (256x256 = 1)")
        ax.legend(loc="upper left", frameon=True)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(FIG_DIR / "fig_interteam_gpu_scaling_float32.pdf")
    plt.close(fig)


def _plot_vs_equipo1_scaling(
    data: InterteamData,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharey=True)
    x = list(range(len(RESOLUTION_ORDER)))
    for idx, pipeline in enumerate(PIPELINE_ORDER):
        ax = axes[idx]
        for team in EMBEDDED_TEAM_ORDER:
            ratios = [
                data[team][pipeline][CUDA][resolution]
                / data[EQUIPO1][pipeline][CUDA][resolution]
                for resolution in RESOLUTION_ORDER
            ]
            ax.plot(
                x,
                ratios,
                marker="o",
                linewidth=2.2,
                color=TEAM_COLORS[team],
                label=team,
            )

        ax.set_title(PIPELINE_LABELS[pipeline])
        ax.set_xticks(x)
        ax.set_xticklabels(RESOLUTION_ORDER)
        ax.set_xlabel("Resolución")
        if idx == 0:
            ax.set_ylabel("Latencia relativa frente al Equipo 1")
        ax.legend(loc="upper left", frameon=True)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(FIG_DIR / "fig_interteam_vs_equipo1_scaling_float32.pdf")
    plt.close(fig)


def main() -> None:
    _ensure_dirs()
    data = _load_float32()

    _write_latency_fps_table(data)
    _write_normalized_table(data)
    _write_self_scaling_table(data)

    _plot_latency(
        data,
        pipeline=STANDARD_RUN,
        output_name="fig_interteam_latency_standard_linear_float32.pdf",
        log_scale=False,
    )
    _plot_latency(
        data,
        pipeline=STANDARD_RUN,
        output_name="fig_interteam_latency_standard_log_float32.pdf",
        log_scale=True,
    )
    _plot_latency(
        data,
        pipeline=COMPLEX_RUN,
        output_name="fig_interteam_latency_complex_linear_float32.pdf",
        log_scale=False,
    )
    _plot_latency(
        data,
        pipeline=COMPLEX_RUN,
        output_name="fig_interteam_latency_complex_log_float32.pdf",
        log_scale=True,
    )
    _plot_fps_comparison(
        data,
        output_name="fig_interteam_fps_comparison_linear_float32.pdf",
        log_scale=False,
    )
    _plot_fps_comparison(
        data,
        output_name="fig_interteam_fps_comparison_log_float32.pdf",
        log_scale=True,
    )
    _plot_embedded_fps_comparison(data)
    _plot_vs_equipo1_scaling(data)
    _plot_gpu_scaling(data)

    print(f"Assets generated in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
