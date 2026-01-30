# TFM [App] — _Pending Name_

Small desktop application for image processing tasks and transformations. Graphical interface built with PyQt6.

**Requirements**

- Python: developed with `3.13` version (compatible with `3.10+`).
- Packages: install from `requirements.txt`.
- (Optional for GPU) Corresponding CUDA drivers in order to use PyTorch with GPU support.

**Quick installation**

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Upgrade `pip` and install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**How to run**

- Run the application:

```bash
python app.py
```

- Run in development mode with automatic reload (hot-reload):

```bash
python run.py
```

**Basic project structure**

- `benchmark/`: operation pipelines benchmarking with pytest.
- `core/`: core image processing package (debayering, filters, format conversion, image I/O).
- `gui/`: PyQt6 GUI components, windows, custom widgets and custom components.
- `images/`: sample images for testing and demonstration.
- `loaders/`: image file loaders supporting different formats.
- `models/`: data models and enumerations for image representation and processing parameters.
- `utils/`: general utility functions and helpers.
- `tests/`: unit test suite for image operations.
- `app.py`: PyQt6 GUI application entry point.
- `run.py`: development script that launches `app.py` and restarts on changes.
- `controller.py`: controller managing application logic and interactions between GUI and core.
