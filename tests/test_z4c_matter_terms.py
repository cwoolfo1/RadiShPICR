import jax.numpy as jnp

from RadiShPICR.Z4C.energy_momentum_tensor import initialize_vacuum_matter_terms
from RadiShPICR.Z4C.z4c_metric import Z4C_Metric


def test_initialize_vacuum_matter_terms_matches_metric_grid():
    r = jnp.linspace(0.1, 1.0, 8)
    zeros = jnp.zeros_like(r)

    metric = Z4C_Metric(
        alpha=jnp.ones_like(r),
        beta=zeros,
        conformal_grr=jnp.ones_like(r),
        conformal_gt=jnp.ones_like(r),
        chi=jnp.ones_like(r),
        Kh=zeros,
        Arr=zeros,
        At=zeros,
        theta=zeros,
        Zr=zeros,
        Gamma=zeros,
        kappa=zeros,
        eta=zeros,
        r=r,
        dr=r[1] - r[0],
    )

    matter_terms = initialize_vacuum_matter_terms(metric)

    assert matter_terms.rho.shape == r.shape
    assert matter_terms.Srr.shape == r.shape
    assert matter_terms.Stt.shape == r.shape
    assert matter_terms.Sr.shape == r.shape
    assert matter_terms.St.shape == r.shape

    assert jnp.allclose(matter_terms.rho, 0.0)
    assert jnp.allclose(matter_terms.Srr, 0.0)
    assert jnp.allclose(matter_terms.Stt, 0.0)
    assert jnp.allclose(matter_terms.Sr, 0.0)
    assert jnp.allclose(matter_terms.St, 0.0)
