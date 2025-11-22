def load_stylesheet(file_path: str) -> str:
    """Read and return the contents of a QSS file.

    Args:
        file_path (str): Path to the QSS file.

    Returns:
        str: Contents of the QSS file.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Could not load stylesheet: {e}")
        return ""
