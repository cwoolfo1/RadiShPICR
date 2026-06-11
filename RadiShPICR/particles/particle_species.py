import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class


class particle_species:
    def __init__(self, name, charge, mass, weight, r, ur, phi, uphi, shape_mode):
        self.name = name
        self.charges = charge
        self.masses = mass
        self.weight = weight
        self.r = r
        self.ur = ur
        self.phi = phi
        self.uphi = uphi
        self.shape_mode = shape_mode

    def get_positions(self):
        return self.r, self.phi
    
    def get_velocities(self):
        return self.ur, self.uphi
    
    def get_mass(self):
        return self.masses * self.weight
    
    def get_charge(self):
        return self.charges * self.weight
    
    def get_shape(self):
        return self.shape_mode
