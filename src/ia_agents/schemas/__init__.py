"""Project data models and validation contracts."""

from .support import SupportQuery

SCHEMAS = {"support": SupportQuery}

__all__ = ["SupportQuery", "SCHEMAS"]
