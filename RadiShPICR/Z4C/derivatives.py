import jax.numpy as jnp

def first_derivative(field, dr):
    # compute 4th order finite difference derivative
    # assume total periodic domain for now

    return (jnp.roll(field, -2) - 8 * jnp.roll(field, -1) + 8 * jnp.roll(field, 1) - jnp.roll(field, 2)) / (12 * dr)


def second_derivative(field, dr):
    # compute 4th order finite difference derivative
    # assume total periodic domain for now

    return (-jnp.roll(field, -2) + 16 * jnp.roll(field, -1) - 30 * field + 16 * jnp.roll(field, 1) - jnp.roll(field, 2)) / (12 * dr ** 2)