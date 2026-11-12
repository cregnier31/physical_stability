"""Physical stability diagnostics for NEMO and AI ocean model fields."""
from .diagnostics import compute_n2, compare_n2, summarize_n2

__all__ = ["compute_n2", "compare_n2", "summarize_n2"]
