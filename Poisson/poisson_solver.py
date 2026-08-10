"""
Finite element solver for the Poisson equation using the Firedrake package.

Description:
This code solves the two-dimensional Poisson equation on a unit square domain
with Dirichlet boundary conditions using the finite element method. The Poisson
equation is given by:
    -Δu = f in Ω
    u = g on ∂Ω
where Ω is the unit square domain, f is a source term, and g is the boundary condition.
We use the method of manufactured solutions to verify the accuracy of the solver.

Author: Divij Ghose
Date: November 2025

"""
from firedrake import *
import matplotlib.pyplot as plt
# from firedrake.pyplot import tricontour

# Define constant parameters
omega = 2 * pi # Frequency for the manufactured solution
num_cells = 100 # Number of cells in each direction for the mesh
checkpointing = False # Checkpointing flag
# Create a unit square mesh
mesh = UnitSquareMesh(num_cells, num_cells)
if checkpointing:
    with CheckpointFile("poisson_example.h5", "w") as chk:
        chk.save_mesh(mesh)
        

# Define function space
V = FunctionSpace(mesh, "CG", 1)

# Define a test function and trial function
u = TrialFunction(V)
v = TestFunction(V)

# Define the manufactured solution and corresponding source term
u_analytical = Function(V)
x, y = SpatialCoordinate(mesh)
u_analytical.interpolate(-sin(omega * x) * sin(omega * y))
f = Function(V)
f.interpolate(-2 * (omega**2) * sin(omega * x) * sin(omega * y))

# Define the variational problem
a = inner(grad(u), grad(v)) * dx
L = f * v * dx

# Define boundary condition
# Note: The manufactured solution is zero on the boundary of the unit square.
bcs = [DirichletBC(V, Constant(0.0), (1, 2, 3, 4))]



# Solve the variational problem
u_solution = Function(V)
solve(a == L, u_solution, bcs=bcs, solver_parameters={'ksp_type': 'cg', 'pc_type': 'ilu'})

# Compute the error between the numerical and analytical solutions
error_L2 = errornorm(u_analytical, u_solution, norm_type='L2')
print(f"L2 Error: {error_L2}")

# # Plot the numerical solution, the analytical solution, and the point-wise absolute error in one figure
fig, axes = plt.subplots(1, 3, figsize=(11, 4))
contours1 = tricontourf(u_solution, axes=axes[0], levels=100, cmap="viridis")
axes[0].set_title("Numerical Solution")
axes[0].set_aspect('equal')
fig.colorbar(contours1, ax=axes[0], shrink=0.625)
contours2 = tricontourf(u_analytical, axes=axes[1], levels=100, cmap="viridis")
axes[1].set_title("Analytical Solution")
axes[1].set_aspect('equal')
fig.colorbar(contours2, ax=axes[1], shrink=0.625)
error_function = Function(V)
error_function.interpolate(abs(u_analytical - u_solution))
contours3 = tricontourf(error_function, axes=axes[2], levels=100, cmap="viridis")
axes[2].set_title("Point-wise Absolute Error")
axes[2].set_aspect('equal')
fig.colorbar(contours3, ax=axes[2], shrink=0.625)
plt.tight_layout()
plt.savefig("poisson_solution.png")
