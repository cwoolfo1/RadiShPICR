## RadiShPICR ##

RadiShPICR is a spherically symmetric particle in cell code that fuses a purely radial electrostatic 
particle-in-cell method with a spherically symmetric formulation of numerical relativity.

This code uses the Kerr-Schild coordinate system for better numerical stability. Particles are evolved using 
a 4th order Runge-Kutta scheme and particle deposition and interpolation is performed using second order shape 
factors.

The relativity implementations are localized by formulation.  Constraint-based
radial metric solves and particle timestepping are imported from
`RadiShPICR.ConstraintBasedRelativity`; Z4C metric evolution helpers are
imported from `RadiShPICR.Z4C`.  `RadiShPICR.evolve` remains as a compatibility
import for the constraint-based `step` and `step_rk4` routines.
