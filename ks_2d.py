"""Solver for the Kuramoto Sivashinsky equation in 1D, given by

    .. math::
        :nowrap:
        \\begin{eqnarray}
        \\partial_t u + \\partial_x^2 u + \\partial_x^4 u u \\partial_x u = 0, \\qquad \\text{in} \\qquad \\Omega \\times (0, T]

        u(x, 0) = u_0(x), \\qquad \\text{in} \\qquad \\Omega
        \\end{eqnarray}
"""

from firedrake import *
from firedrake.adjoint import *
from pyadjoint import *
import matplotlib.pyplot as plt
import numpy as np
N = 30
L = 32.0 
dt = 1e-4
t_end = 5
mesh = PeriodicIntervalMesh(N, L)
V = FunctionSpace(mesh, "CG", 2)
W = V * V
uw = Function(W)
u, w = split(uw)

plot_function_space = FunctionSpace(mesh, "CG", 1)

uw_n = Function(W)
u_n, w_n = split(uw_n)

v, phi = TestFunctions(W)
x = SpatialCoordinate(mesh)

init_expr = cos(x[0] / 16.0) * (1.0 + sin(x[0] / 16.0)) + 0.5 * cos(x[0] / 8.0) * (1.0 + sin(x[0] / 8.0))
u_init = Function(V, name="Initial condition")
u_init.interpolate(init_expr)
w_init = Function(V, name="Initial condition")
phi_init =TestFunction(V)
w_trial = TrialFunction(V)

a_w = w_trial * phi_init * dx
L_w = +inner(grad(u_init), grad(phi_init)) * dx

solver_w = LinearVariationalSolver(LinearVariationalProblem(a_w, L_w, w_init))
solver_w.solve()

uw_n.sub(0).assign(u_init)
uw_n.sub(1).assign(w_init)

uw.sub(0).assign(u_init)
uw.sub(1).assign(w_init)

w_mid = 0.5*(w + w_n)
def du_dt(u, u_n, dt):
    return (u - u_n) / dt

mu = Constant(1.0)
# --------------------------
# Variational form
# --------------------------
F = (
    inner((du_dt(u, u_n, dt)), v) *dx
    - 0.5 * inner(u_n**2, v) * dx
    - inner(w_mid, v) * dx
    + mu * inner(grad(w_mid), grad(v)) * dx
    + inner(w, phi) * dx
    - inner(grad(u), grad(phi)) * dx

)

# Jacobian for Newton
J = derivative(F, uw)

# --------------------------
# Nonlinear problem / solver
# --------------------------
problem = NonlinearVariationalProblem(F, uw, J=J)
solver = NonlinearVariationalSolver(
    problem,
solver_parameters={"snes_monitor": None,
                             "ksp_type": "gmres",
                             "mat_type": "aij",
                             "pc_type": "lu",
                             "pc_factor_mat_solver_type": "mumps"}
)


# problem = LinearVariationalProblem(lhs(F), rhs(F), )
# solver = LinearVariationalSolver(
#     problem,
#     solver_parameters={
#         "ksp_type": "preonly",
#         "pc_type": "lu"
#     }
# )
# --------------------------
# Output
# --------------------------
outfile = VTKFile("ks_mixed.pvd")

u_out, w_out = uw.subfunctions
u_out.rename("u")
w_out.rename("w")
outfile.write(u_out, w_out, time=0.0)

# --------------------------
# Time loop
# --------------------------
t = 0.0
step = 0
u_data = []
t_data = []
# u_plot = project(u_out, plot_function_space)
# t_data.append(t)

while t < t_end:
    
    uw.assign(uw_n)
    solver.solve()
    # Update previous solution
    uw_n.assign(uw)
    t += dt
    t_data.append(t)
    step += 1

    u_out, w_out = uw.subfunctions
    outfile.write(u_out, w_out, time=t)
    u_plot = project(u_out, plot_function_space)
    u_data.append(u_plot.dat.data[:])

    print(f"step = {step}, t = {t:.5f}")

fig, ax = plt.subplots()
#2d contour plot of u(x,t)
t_data = np.array(t_data)
u_data = np.array(u_data)
X, T = np.meshgrid(np.linspace(0, L, N), t_data)
c = ax.contourf(T, X, u_data, levels=50, cmap='viridis')
ax.set_xlabel('T')
ax.set_ylabel('x')
ax.set_title('Kuramoto Sivashinsky equation')
fig.colorbar(c, ax=ax)
plt.savefig('ks_mixed.png', dpi=300)