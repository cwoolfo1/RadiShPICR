## Solver Logic For Poisson Equation ##


$$

\partial_1 F^{1 0} + \Gamma^1_{1 1} F^{1 0} => \partial_r E_r + \partial_r \ln (A) E_r = \frac{\rho}{\epsilon_0}

$$

That is the relativistic Poisson equation in the curved spacetime for a spherically symmetric relativistic system.

We will use a finite difference discretization of the r derivative to formulate this equation as a matrix inversion 
problem. We will use a uniform grid in r, with grid points $r_i = i \Delta r$ for $i = 0, 1, ..., N-1$, where $\Delta r$ is the grid spacing. Just solving the interior points for now.


$$

\frac{E_{r,i+1} - E_{r,i-1}}{2 \Delta r} + \frac{1}{A_i} \frac{A_{i+1} - A_{i-1}}{2 \Delta r} E_{r,i} = \frac{\rho_i}{\epsilon_0}

$$

Due to the diagonal elements of A, there is no guarantee that the matrix will be symmetric positive definite. So, 
use GMRES to solve the the matrix inversion problem.