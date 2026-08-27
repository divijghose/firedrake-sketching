# Firedrake Sketching
## Poisson Equation
[poisson_solver.py](Poisson/poisson_solver.py) is a simple example of solving the Poisson equation using Firedrake. It demonstrates how to set up the problem, define the function spaces, and solve the equation using finite element methods.
[poisson_solver_multigrid.py](Poisson/poisson_solver_multigrid.py) is an extension of the previous example with a mesh hierarchy.
## Navier-Stokes Equations
[tnse2d_lid_driven_cavity.py](TNSE2D/tnse2d_lid_driven_cavity.py) is an example of solving the time-dependent 2D Navier-Stokes equations for a lid-driven cavity flow problem.
[tnse2d_flow_past_cylinder.py](TNSE2D/tnse2d_flow_past_cylinder.py) is an example of solving the time-dependent 2D Navier-Stokes equations for flow past a cylinder.
## Kuramoto-Sivashinsky Equation
[ks_1d_c0ip.py](TKS/ks_1d_c0ip.py) is an example of solving the 1D Kuramoto-Sivashinsky equation using a C0 interior penalty method.
[ks_2d_c0ip.py](TKS/ks_2d_c0ip.py) is an example of solving the 2D Kuramoto-Sivashinsky equation using a C0 interior penalty method.
## Cahn-Hilliard Equation
[cahn-hillard.py](TCH/cahn-hillard.py) is an example of solving the Cahn-Hillard equation with an auxillary variable.