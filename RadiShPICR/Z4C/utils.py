import jax.numpy as jnp
from jax import jit

from RadiShPICR.Z4C.z4c_metric import Z4C_Metric

def trace_free_curvature(Arr, At, metric: Z4C_Metric):
    conformal_grr = metric.conformal_grr
    conformal_gt = metric.conformal_gt
    chi = metric.chi
    # unpack the metric components

    trace = Arr / conformal_grr + 2.0 * At / conformal_gt
    # compute the trace of the curvature

    Arr_trace_free = Arr - (1.0 / 3.0) * conformal_grr * trace
    At_trace_free = At - (1.0 / 3.0) * conformal_gt * trace
    # compute the trace-free parts

    return Arr_trace_free, At_trace_free

def generate_r_grid(r_min, r_max, Nr):
    r_grid = jnp.linspace(r_min, r_max, Nr)
    r_grid += (r_grid[1] - r_grid[0]) / (2 * Nr)  # shift cells so cell center is at dr/2
    return r_grid
