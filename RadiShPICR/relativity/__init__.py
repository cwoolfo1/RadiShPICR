"""Spherically symmetric relativity tools for RadiShPICR."""

from RadiShPICR.relativity.grid import RadialGrid, build_radial_grid

__all__ = [
    "MetricState",
    "RadialGrid",
    "build_radial_grid",
    "compute_metric",
]


def __getattr__(name):
    if name in {"MetricState", "compute_metric"}:
        from RadiShPICR.relativity.metric import MetricState, compute_metric

        return {"MetricState": MetricState, "compute_metric": compute_metric}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
