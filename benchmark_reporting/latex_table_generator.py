import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import argparse


def load_benchmark_json(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r") as f:
        return json.load(f)


def extract_benchmark_data(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    results = []

    for benchmark in data.get("benchmarks", []):
        params = benchmark.get("params", {})
        device = params.get("device", "N/A")
        shape = params.get("shape", [])
        dtype_str = str(params.get("dtype", "N/A"))

        if "float16" in dtype_str:
            dtype = "float16"
        elif "float32" in dtype_str:
            dtype = "float32"
        elif "float64" in dtype_str:
            dtype = "float64"
        else:
            dtype = "unknown"

        if isinstance(shape, list) and len(shape) >= 3:
            shape_str = f"{shape[1]}x{shape[2]}"
        else:
            shape_str = str(shape)

        stats = benchmark.get("stats", {})

        result = {
            "Device": device,
            "Shape": shape_str,
            "Dtype": dtype,
            "Min (s)": stats.get("min", 0),
            "Max (s)": stats.get("max", 0),
            "Mean (s)": stats.get("mean", 0),
            "Median (s)": stats.get("median", 0),
            "Stddev (s)": stats.get("stddev", 0),
            "Quartile 1 (s)": stats.get("q1", 0),
            "Quartile 3 (s)": stats.get("q3", 0),
            "Ops/s": stats.get("ops", 0),
            "Rounds": stats.get("rounds", 0),
            "Total (s)": stats.get("total", 0),
        }

        results.append(result)

    return results


def save_to_csv(df: pd.DataFrame, output_path: str):
    df.to_csv(output_path, index=False, float_format="%.6f")
    print(f"CSV saved at: {output_path}")


def generate_latex_table(
    df: pd.DataFrame,
    output_path: str,
    caption: str = "Resultados del Benchmark",
    label: str = "tab:benchmark",
):
    df_latex = df[
        ["Device", "Shape", "Dtype", "Mean (s)", "Median (s)", "Stddev (s)", "Ops/s"]
    ].copy()

    df_latex["Mean (s)"] = df_latex["Mean (s)"].apply(lambda x: f"{x:.6f}")
    df_latex["Median (s)"] = df_latex["Median (s)"].apply(lambda x: f"{x:.6f}")
    df_latex["Stddev (s)"] = df_latex["Stddev (s)"].apply(lambda x: f"{x:.6f}")
    df_latex["Ops/s"] = df_latex["Ops/s"].apply(lambda x: f"{x:.2f}")
    df_latex["Dtype"] = df_latex["Dtype"].apply(lambda x: f"\\textit{{{x}}}")

    latex_table = df_latex.to_latex(
        index=False,
        escape=False,
        column_format="l" * len(df_latex.columns),
        caption=caption,
        label=label,
    )

    latex_str = (
        "\\documentclass{article}\n"
        "\\usepackage{booktabs}\n"
        "\\begin{document}\n\n"
        f"{latex_table}"
        "\\end{document}\n"
    )

    with open(output_path, "w") as f:
        f.write(latex_str)

    print(f"LaTeX table saved at: {output_path}")


def generate_latex_table_by_device(df: pd.DataFrame, output_dir: str):
    devices = df["Device"].unique()

    for device in devices:
        df_device = df[df["Device"] == device].copy()

        df_latex = df_device[
            ["Shape", "Dtype", "Mean (s)", "Stddev (s)", "Ops/s", "Rounds"]
        ].copy()

        df_latex["Mean (s)"] = df_latex["Mean (s)"].apply(lambda x: f"{x:.6f}")
        df_latex["Stddev (s)"] = df_latex["Stddev (s)"].apply(lambda x: f"{x:.6f}")
        df_latex["Ops/s"] = df_latex["Ops/s"].apply(lambda x: f"{x:.2f}")
        df_latex["Dtype"] = df_latex["Dtype"].apply(lambda x: f"\\textit{{{x}}}")

        df_latex.columns = [
            "Shape",
            "Dtype",
            "Mean (s)",
            "Std Dev (s)",
            "Ops/s",
            "Rounds",
        ]

        caption = f"Resultados del Benchmark - {device.upper()}"
        label = f"tab:benchmark_{device}"

        latex_table = df_latex.to_latex(
            index=False,
            escape=False,
            column_format="llrrrr",
            caption=caption,
            label=label,
        )

        latex_str = (
            "\\documentclass{article}\n"
            "\\usepackage{booktabs}\n"
            "\\begin{document}\n\n"
            f"{latex_table}"
            "\\end{document}\n"
        )

        output_path = Path(output_dir) / f"benchmark_{device}.tex"
        with open(output_path, "w") as f:
            f.write(latex_str)

        print(f"LaTeX table for {device} saved at: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert benchmark results from JSON to CSV and LaTeX"
    )
    parser.add_argument("input_json", help="Path to the input JSON file")
    parser.add_argument(
        "--split-by-device",
        action="store_true",
        help="Generate separate LaTeX tables by device",
    )

    args = parser.parse_args()

    output_dir = Path(Path(args.input_json).stem)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from: {args.input_json}")
    data = load_benchmark_json(args.input_json)

    results = extract_benchmark_data(data)
    df = pd.DataFrame(results)

    print(f"\nTotal benchmarks: {len(df)}")
    print(f"Shapes: {df['Shape'].unique()}")
    print(f"Devices: {df['Device'].unique()}")
    print(f"Dtypes: {df['Dtype'].unique()}")

    csv_path = output_dir / f"{Path(args.input_json).stem}.csv"
    save_to_csv(df, str(csv_path))

    latex_path = output_dir / f"{Path(args.input_json).stem}.tex"
    generate_latex_table(df, str(latex_path))

    if args.split_by_device:
        generate_latex_table_by_device(df, str(output_dir))


if __name__ == "__main__":
    main()
