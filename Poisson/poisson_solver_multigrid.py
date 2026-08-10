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
from firedrake.pyplot import tricontour

# Define constant parameters
omega = 16 * pi # Frequency for the manufactured solution
init_num_cells = 10 # Number of cells in each direction for the mesh
mesh_levels = 4
checkpointing = False # Checkpointing flag

# Coarse mesh solver
# Create a unit square mesh
coarse_mesh = UnitSquareMesh(init_num_cells, init_num_cells)
V_coarse = FunctionSpace(coarse_mesh, "CG", 1)
u_coarse = TrialFunction(V_coarse)
v_coarse = TestFunction(V_coarse)
u_analytical_coarse = Function(V_coarse)
x_coarse, y_coarse = SpatialCoordinate(coarse_mesh)
u_analytical_coarse.interpolate(-sin(omega * x_coarse) * sin(omega * y_coarse))
f_coarse = Function(V_coarse)
f_coarse.interpolate(-2 * (omega**2) * sin(omega * x_coarse) * sin(omega * y_coarse))
a_coarse = inner(grad(u_coarse), grad(v_coarse)) * dx
L_coarse = f_coarse * v_coarse * dx
bcs = [DirichletBC(V_coarse, Constant(0.0), (1, 2, 3, 4))]
u_solution_coarse = Function(V_coarse)
solve(a_coarse == L_coarse, u_solution_coarse, bcs=bcs, solver_parameters={'ksp_type': 'cg', 'pc_type': 'ilu'})

# Find error on coarse mesh
error_L2_coarse = errornorm(u_analytical_coarse, u_solution_coarse, norm_type='L2')
print(f"Coarse Mesh L2 Error: {error_L2_coarse}")

# Define a mesh hierarchy for multigrid
hierarchy = MeshHierarchy(coarse_mesh, mesh_levels)



# Set up the problem on the finest mesh
mesh = hierarchy[-1]
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
bcs = [DirichletBC(V, zero(), (1, 2, 3, 4))] # Same as [DirichletBC(V, Constant(0.0), (1, 2, 3, 4))]



# Solve the variational problem
u_solution = Function(V)
solve(a == L, u_solution, bcs=bcs, solver_parameters={'ksp_type': 'cg', 'pc_type': 'ilu'})

# Interpolate coarse solution to fine mesh for comparison
u_coarse_to_fine = Function(V)
u_coarse_to_fine.interpolate(u_solution_coarse)


# Compute the error between the numerical and analytical solutions
error_L2 = errornorm(u_analytical, u_solution, norm_type='L2')
print(f"L2 Error: {error_L2}")

# Plot the numerical solution, the analytical solution, and the point-wise absolute error in one figure
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
contours1 = tricontourf(u_solution, axes=axes[0][0], levels=100, cmap="viridis")
axes[0][0].set_title("Numerical Solution Multigrid")
fig.colorbar(contours1, ax=axes[0][0])
contours2 = tricontourf(u_coarse_to_fine, axes=axes[1][0], levels=100, cmap="viridis")
axes[1][0].set_title("Numerical Solution Coarse Interpolated")
fig.colorbar(contours2, ax=axes[1][0])
contours3 = tricontourf(u_analytical, axes=axes[0][1], levels=100, cmap="viridis")
axes[0][1].set_title("Analytical Solution")
fig.colorbar(contours3, ax=axes[0][1])
error_function = Function(V)
error_function.interpolate(abs(u_analytical - u_solution))
contours4 = tricontourf(error_function, axes=axes[1][1], levels=100, cmap="viridis")
axes[1][1].set_title("Point-wise Absolute Error")
fig.colorbar(contours3, ax=axes[1][1])
plt.tight_layout()
plt.savefig("poisson_solutions_comparison.png")
