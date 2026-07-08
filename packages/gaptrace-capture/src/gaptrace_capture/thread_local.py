import threading

_local = threading.local()


def set_active_capture(cap) -> None:
    """Make ``cap`` this thread's active Capture. Pure (thread-local only)."""
    _local.capture = cap


def get_active_capture():
    """Return this thread's active Capture, or None. Pure."""
    return getattr(_local, "capture", None)


def clear_active_capture() -> None:
    """Drop this thread's active Capture, if any. Pure."""
    if hasattr(_local, "capture"):
        del _local.capture
