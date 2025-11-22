import logging
import traceback
from typing import Any, Callable

from utils.error import show_error


def safe_call(
    func: Callable[..., Any],
    *args: Any,
    title: str = "Error",
    re_raise: bool = False,
    fallback: Any | None = None,
    show: bool = True,
    **kwargs: Any,
) -> Any:
    """Call `func(*args, **kwargs)` catching exceptions.

    On exception, the error is presented via `show_error` (if `show=True`) and
    also logged. Returns `fallback` (default None) unless `re_raise` is True.

    Args:
        func (Callable[..., Any]): _function to call_.
        *args (Any): _positional arguments for func_.
        **kwargs (Any): _keyword arguments for func_.
        title (str, optional): _title for error dialog_. Defaults to "Error".
        re_raise (bool, optional): _whether to re-raise the exception_. Defaults to False.
        fallback (Any | None, optional): _value to return if an exception occurs_. Defaults to None.
        show (bool, optional): _whether to show the error dialog_. Defaults to True.

    Returns:
        Any: _result of the function call or fallback value_
    """
    try:
        return func(*args, **kwargs)
    except (
        Exception
    ) as exc:  # pragma: no cover - behaviour exercised in manual runs/tests
        tb = traceback.format_exc()
        msg = str(exc)

        try:
            arg_types = ",".join([type(a).__name__ for a in args]) if args else ""
            kw_keys = ",".join(list(kwargs.keys())) if kwargs else ""
            ctx = f"func={getattr(func, '__name__', repr(func))} args=[{arg_types}] kwargs=[{kw_keys}]"
        except Exception:
            ctx = f"func={getattr(func, '__name__', repr(func))}"

        if show:
            try:
                show_error(title, msg, detailed=tb)
            except Exception:
                logging.exception("show_error failed inside safe_call — %s", ctx)

        if re_raise:
            raise

        return fallback
