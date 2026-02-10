# Benchmark Reporting

## Latex table generator

Script to generate a latex table from the results of the experiments.

1. Reads the results from JSON file.
2. Converts the results into CSV format.
3. Generates a Pandas DataFrame from the CSV data.
4. Creates a LaTeX table from the DataFrame.

---

Usage:

```bash
python latex_tanble_generator.py <results_file.json>
```

Optional arguments:

- `--split-by-device`: generate separate LaTex tables by device

---

<br>

Results will be saved in a directory with the same name as the results file. Example:

```
└── ./
    ├── 001_run.json
    └── 001_run/
        ├── 001_run.csv
        └── 001_run.tex
```
