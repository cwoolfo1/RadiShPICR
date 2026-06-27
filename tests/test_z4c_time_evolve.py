import jax.numpy as jnp

from RadiShPICR.Z4C.energy_momentum_tensor import initialize_vacuum_matter_terms
from RadiShPICR.Z4C.z4c_metric import Z4C_Metric


def _flat_metric(r):
    zeros = jnp.zeros_like(r)
    ones = jnp.ones_like(r)

    return Z4C_Metric(
        alpha=ones,
        beta=zeros,
        conformal_grr=ones,
        conformal_gt=ones,
        chi=ones,
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


def test_metric_time_derivatives_match_metric_layout():
    from RadiShPICR.Z4C.time_evolve import metric_time_derivatives

    r = jnp.linspace(0.1, 1.0, 16)
    metric = _flat_metric(r)
    matter_terms = initialize_vacuum_matter_terms(metric)

    derivatives = metric_time_derivatives(metric, matter_terms)

    for derivative_field, metric_field in zip(derivatives, metric):
        assert jnp.shape(derivative_field) == jnp.shape(metric_field)

    assert jnp.allclose(derivatives.kappa, 0.0)
    assert jnp.allclose(derivatives.eta, 0.0)
    assert jnp.allclose(derivatives.r, 0.0)
    assert jnp.allclose(derivatives.dr, 0.0)


def test_rk4_step_keeps_grid_and_damping_parameters_fixed():
    from RadiShPICR.Z4C.time_evolve import rk4_step

    r = jnp.linspace(0.1, 1.0, 16)
    metric = _flat_metric(r)
    matter_terms = initialize_vacuum_matter_terms(metric)

    updated = rk4_step(metric, matter_terms, dt=1.0e-3)

    assert jnp.allclose(updated.kappa, metric.kappa)
    assert jnp.allclose(updated.eta, metric.eta)
    assert jnp.allclose(updated.r, metric.r)
    assert jnp.allclose(updated.dr, metric.dr)


def test_rk4_step_preserves_flat_vacuum_metric():
    from RadiShPICR.Z4C.time_evolve import rk4_step

    r = jnp.linspace(0.1, 1.0, 16)
    metric = _flat_metric(r)
    matter_terms = initialize_vacuum_matter_terms(metric)

    updated = rk4_step(metric, matter_terms, dt=1.0e-3)

    for updated_field, metric_field in zip(updated, metric):
        assert jnp.allclose(updated_field, metric_field)


def test_rk4_step_uses_classic_stage_weights(monkeypatch):
    import RadiShPICR.Z4C.time_evolve as time_evolve

    r = jnp.linspace(0.1, 1.0, 8)
    metric = _flat_metric(r)
    matter_terms = initialize_vacuum_matter_terms(metric)
    stage_values = [1.0, 2.0, 3.0, 4.0]

    def fake_metric_time_derivatives(stage_metric, stage_matter_terms):
        stage_value = stage_values.pop(0)
        zeros = jnp.zeros_like(stage_metric.r)
        derivative = jnp.full_like(stage_metric.alpha, stage_value)

        return Z4C_Metric(
            alpha=derivative,
            beta=zeros,
            conformal_grr=zeros,
            conformal_gt=zeros,
            chi=zeros,
            Kh=zeros,
            Arr=zeros,
            At=zeros,
            theta=zeros,
            Zr=zeros,
            Gamma=zeros,
            kappa=zeros,
            eta=zeros,
            r=zeros,
            dr=jnp.asarray(0.0, dtype=stage_metric.dr.dtype),
        )

    monkeypatch.setattr(
        time_evolve,
        "metric_time_derivatives",
        fake_metric_time_derivatives,
    )

    updated = time_evolve.rk4_step(metric, matter_terms, dt=0.6)

    expected_alpha = metric.alpha + 0.6 * (1.0 + 2.0 * 2.0 + 2.0 * 3.0 + 4.0) / 6.0
    assert jnp.allclose(updated.alpha, expected_alpha)
    assert jnp.allclose(updated.beta, metric.beta)
    assert jnp.allclose(updated.r, metric.r)
    assert stage_values == []
