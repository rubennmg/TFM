import logging
import traceback


def show_error(title: str, message: str, detailed: str | None = None) -> None:
    """Present an error to the user.

    Args:
        title (str): Title of the error.
        message (str): Message describing the error.
        detailed (str | None, optional): Detailed error information. Defaults to None.
    """
    if detailed is None:
        detailed = traceback.format_exc()

    # log to stderr
    try:
        import sys

        if detailed:
            print(f"{title}: {message}\n{detailed}", file=sys.stderr)
        else:
            print(f"{title}: {message}", file=sys.stderr)
    except Exception:
        logging.exception("Failed printing error to stderr")

    # show GUI dialog
    try:
        from gui.helpers.error_dialog import show_error_dialog

        try:
            show_error_dialog(title, message, detailed=detailed)
        except Exception:
            logging.exception("GUI error dialog raised an exception")
    except Exception:
        logging.debug("GUI error dialog unavailable or import failed")
