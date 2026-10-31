"""Optional, best-effort observation; observers never control agent decisions."""
import logging


def emit(observer, kind, **fields):
    if observer is not None:
        try:
            observer({"kind": kind, **fields})
        except Exception:
            logging.getLogger(__name__).exception("Agent observer failed")
