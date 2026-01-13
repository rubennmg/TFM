# Benchmark

Run benckmark:

```bash
pytest benchmark/run.py --bench-profile=<profile_name>
```

Example usage (on main directory `/TFM/app`):

```bash
pytest benchmark/run.py --bench-profile=pipelines/pipeline_1.json
```

Results will be saved in `.benchmarks/<profile_name>.json`.
