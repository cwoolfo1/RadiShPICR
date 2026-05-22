def test_radial_electric_solver_imports_without_relativity_metric_cycle():
    from RadiShPICR.EM import solve_radial_electric_field

    assert callable(solve_radial_electric_field)
