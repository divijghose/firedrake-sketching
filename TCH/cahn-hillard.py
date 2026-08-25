from firedrake import *
import numpy as np
import matplotlib.pyplot as plt
import os
from firedrake.ufl_expr import ufl
import sys

# Read configuration from YAML file
if len(sys.argv) > 1 and sys.argv[1].endswith(".yaml"):
    PETSc.Sys.Print(f"Reading configuration from YAML file: {sys.argv[1]}")
    yaml_file_path = sys.argv[1]
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required to read YAML files.")
    with open(yaml_file_path, "r") as file:
        config = yaml.safe_load(file)
else:
    raise ValueError("Please provide a YAML configuration file as a command-line argument.")

os.makedirs("results", exist_ok=True)
output_dir = config.get("output_dir", "ch_output")
output_dir = f"results/{output_dir}"
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
outfile = VTKFile(f"{output_dir}/tch.pvd")

N = config.get("num_elems", 100)
mesh = UnitSquareMesh(N, N)

lmbda = float(config.get("lambda", 1e-2))
dt = float(config.get("dt", 5e-6))
theta = float(config.get("theta", 0.5))
t = 0.0
T = 50*dt
tdump = config.get("output_frequency", 10) * dt
dumpt = 0.


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

mu_mid = (1 - theta) * mu0 + theta * mu # Crank-Nicholson for theta = 0.5

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
solver_parameters = {
    "snes_type": "newtonls",
    "snes_rtol": 1e-50,
    "snes_atol": 1e-50,
    "snes_stol": np.sqrt(np.finfo(np.float64).eps) * 1e-2,
    "ksp_type": "preonly",
    "snes_linesearch_type": "none",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}
if config.get("verbose", False):
    solver_parameters["snes_monitor"] = None

problem = NonlinearVariationalProblem(F, u)
solver = NonlinearVariationalSolver(problem, solver_parameters=solver_parameters)


c_plot, mu_plot = u.subfunctions
outfile.write(c_plot)
u0.assign(u)

while t < T:
    t += dt
    if config.get("verbose", True):
        PETSc.Sys.Print(f"t = {t}")
    solver.solve()
    u0.assign(u)


    if dumpt > tdump - dt/2:
        c_plot, mu_plot = u.subfunctions
        outfile.write(c_plot)
        dumpt -= tdump
    dumpt += dt
