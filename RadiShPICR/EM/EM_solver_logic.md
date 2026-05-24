## Solver Logic For Radial Gauss Law ##


$$

\frac{1}{r^2 A}\partial_r \left(r^2 A E_r\right) => \partial_r E_r + \left(\frac{2}{r} + \partial_r \ln (A)\right) E_r = \frac{\rho}{\epsilon_0}

$$

That is radial Gauss law in the curved spacetime for a spherically symmetric relativistic system.

We will use a finite difference discretization of the r derivative to formulate this equation as a matrix inversion
problem. We will use a uniform grid in r, with grid points $r_i = i \Delta r$ for $i = 0, 1, ..., N-1$, where $\Delta r$ is the grid spacing. The center is fixed by spherical symmetry, and the outer point is solved with a backward finite-difference row.


$$

\frac{E_{r,i+1} - E_{r,i-1}}{2 \Delta r} + \left(\frac{2}{r_i} + \frac{1}{A_i} \frac{A_{i+1} - A_{i-1}}{2 \Delta r}\right) E_{r,i} = \frac{\rho_i}{\epsilon_0}

$$

Due to the diagonal elements of A, there is no guarantee that the matrix will be symmetric positive definite. The current implementation uses a sparse direct solve for this matrix inversion problem.
