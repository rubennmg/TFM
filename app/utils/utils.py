import torch 

def get_device() -> torch.device:
    """Get the available device (GPU if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_stylesheet(file_path: str) -> str:
    """Read and return the contents of a QSS file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Could not load stylesheet: {e}")
        return ""