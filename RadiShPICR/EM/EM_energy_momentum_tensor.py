import jax.numpy as jnp

def compute_EM_energy_density(Er):
    """
    Compute the electromagnetic energy density.
    
    Calculates the energy density of the electromagnetic field using the radial
    electric field and the lapse function (metric coefficient). The energy density
    is given by: u_EM = alpha^2 / 2 * (Er^2), where alpha is the lapse function.
    
    Parameters
    ----------
    Er : array_like or float
        Radial component of the electric field.
    
    Returns
    -------
    energy_density : array_like or float
        The electromagnetic energy density at each point in space.
        Same shape as the input arrays Er and lapse.
    
    Notes
    -----
    This formula is derived from relativistic electromagnetism where the energy
    density is weighted by the square of the lapse function to account for
    time dilation effects in the spacetime metric.
    
    Examples
    --------
    >>> Er = 1.0
    >>> u_EM = compute_EM_energy_density(Er)
    """
    # u_EM = Er^2 / 2

    energy_density = 0.5 * Er**2
    return energy_density


def compute_EM_stress(Er, A):
    """
    Compute the radial-radial component of the electromagnetic stress tensor.
    
    This function calculates the S_rr component of the Maxwell stress tensor,
    representing the radial electromagnetic stress due to the electric field.
    
    Parameters
    ----------
    Er : float or array-like
        Radial component of the electric field.
    A : float or array-like
        Geometric or normalization factor (typically related to the radial coordinate).
    
    Returns
    -------
    float or ndarray
        The radial-radial stress tensor component: 0.5 * A^2 * Er^2.
    
    Notes
    -----
    The Maxwell stress tensor S_rr is derived from:
    S_rr = A^2 * (Er^2 - 0.5 * Er^2) = 0.5 * A^2 * Er^2
    
    This represents the electromagnetic pressure/tension in the radial direction.
    """

    # S_rr = A^2 * (Er^2 - 0.5 * Er^2) = 0.5 * A^2 * Er^2
    stress = 0.5 * A**2 * Er**2
    return stress