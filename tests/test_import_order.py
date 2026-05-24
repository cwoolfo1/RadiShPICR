def test_radial_electric_solver_imports_without_relativity_metric_cycle():
    from RadiShPICR.EM import compute_charge_density_and_radial_electric_field

    assert callable(compute_charge_density_and_radial_electric_field)
