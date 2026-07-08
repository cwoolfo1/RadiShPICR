import jax.numpy as jnp

def first_derivative(field, dr, parity=-1):
    # compute finite difference derivative
    # reduce to 4th order finite difference in the interior, and use forward/backward difference at the boundaries

    Nr = field.shape[0]
    # get the number of grid points

    field_ = jnp.zeros(shape=(Nr+2)) # initialize dummy array
    field_ = field_.at[2:].set(field) # add field
    field_ = field_.at[:2].set( field_[2:4] * parity ) # reflect the field across the inner boundary with parity value

    derivative  = (-jnp.roll(field_, -2) + 8 * jnp.roll(field_, -1) - 8 * jnp.roll(field_, 1) + jnp.roll(field_, 2)) / (12 * dr)
    # define the fourth order finite difference for the first derivative

    return derivative[2:]

def second_derivative(field, dr, parity=-1):
    # compute finite difference derivative
    # assume total periodic domain for now

    Nr = field.shape[0]
    # get the number of grid points

    field_ = jnp.zeros(shape=(Nr+2)) # initialize dummy array
    field_ = field_.at[2:].set(field) # add field
    field_ = field_.at[:2].set( field_[2:4] * parity ) # reflect the field across the inner boundary with parity value

    derivative = (-jnp.roll(field_, -2) + 16 * jnp.roll(field_, -1) - 30 * field_ + 16 * jnp.roll(field_, 1) - jnp.roll(field_, 2)) / (12 * dr ** 2)
    # define the fourth order finite difference for the second derivative

    return derivative[2:]


def sixth_derivative(field, dr, parity=-1):
    # compute 6th order finite difference derivative
    # assume total periodic domain for now

    Nr = field.shape[0]
    # get the number of grid points

    field_ = jnp.zeros(shape=(Nr+3)) # initialize dummy array
    field_ = field_.at[3:].set(field) # add field
    field_ = field_.at[:3].set( field_[3:6] * parity ) # reflect the field across the inner boundary with parity value

    derivative =  (jnp.roll(field_, -3) - 6 * jnp.roll(field_, -2) + 15 * jnp.roll(field_, -1) - 20 * field_ + 15 * jnp.roll(field_, 1) - 6 * jnp.roll(field_, 2) + jnp.roll(field_, 3)) / (dr ** 6)
    # define the 6th order finite difference for the 6th derivative

    return derivative[3:]
