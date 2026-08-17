from firedrake import *
import numpy as np
import matplotlib.pyplot as plt
import os
from firedrake.ufl_expr import ufl

output_dir = "ch_output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

N = 100
L = 1.0
mesh = UnitSquareMesh(N, N)

lmbda = 1e-2
dt = 5e-6
theta = 0.5
t = 0.0


V = FunctionSpace(mesh, "CG", 1)
W = V * V

q, v = TestFunctions(W)

u = Function(W)
u0 = Function(W)

c0, mu0 = split(u0)
c, mu = split(u)

pcg = PCG64(seed=42)
rg = RandomGenerator(pcg)
u_init = rg.random(V)
u_init.assign(0.5 + 0.01 * (0.5 - u_init))

u.sub(0).assign(u_init)


c = ufl.variable(c)
f = 100 * c**2 * (1 - c)**2
dfdc = ufl.diff(f, c)

mu_mid = (1 - theta) * mu0 + theta * mu 

F0 = (
    inner(c, q) * dx
    - inner(c0, q) * dx
    + dt * inner(grad(mu_mid), grad(q)) * dx
)
F1 = (
    inner(mu, v) * dx
    - inner(dfdc, v) * dx
    - lmbda * inner(grad(c), grad(v)) * dx
)
F = F0 + F1


problem = NonlinearVariationalProblem(F, u)
solver = NonlinearVariationalSolver(problem, solver_parameters={
    "snes_type": "newtonls",
    "snes_rtol": 1e-50,
    "snes_atol": 1e-50,
    "snes_stol": np.sqrt(np.finfo(np.float64).eps) * 1e-2,
    "ksp_type": "preonly",
    "snes_linesearch_type": "none",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "snes_monitor": None,
})
file = VTKFile(f"{output_dir}/tch.pvd")
c_plot, mu_plot = u.subfunctions
file.write(c_plot)
u0.assign(u)
T = 50*dt
while t < T:
    t += dt

    print(f"t = {t}")
    solver.solve()
    u0.assign(u)

    c_plot, mu_plot = u.subfunctions

    file.write(c_plot)