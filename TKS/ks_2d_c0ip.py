
from firedrake import *
from math import pi
import numpy as np
import matplotlib.pyplot as plt
import os 
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
output_dir = config.get("output_dir", "ks_output")
output_dir = f"results/{output_dir}"
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
outfile = VTKFile(f"{output_dir}/tks_2d.pvd")


N = config.get("num_elems", 128)
vx = float(config.get("vx", 0.035))
vy = float(config.get("vy", 0.035))
Lx = np.sqrt(np.pi**2/vx)
Ly = np.sqrt(np.pi**2/vy)
mesh = PeriodicRectangleMesh(nx=N, ny=N, Lx=2*Lx, Ly=2*Ly)
dt = float(config.get("dt", 1e-2))
T = float(config.get("tmax", 100.0))
tdump = float(config.get("output_freq", 10.0)) * dt
dumpt = 0.

alpha = Constant(float(config.get("viscosity", 1.0))) # viscosity
beta = Constant(float(config.get("hyperviscosity", 1.0))) # hyperviscosity
gamma= Constant(float(config.get("advection", 1.0))) # advection
theta = float(config.get("theta", 0.5)) # 0.5 for Crank-Nicolson, 1.0 for backward Euler

V = FunctionSpace(mesh, "CG", 2)
Vdg = FunctionSpace(mesh, "CG", 1)
unp1 = Function(V)
un = Function(V)
v = TestFunction(V)

uh = (1 - theta) * un + theta * unp1
uh_mean = assemble(uh*uh*dx)/(4*Lx*Ly)

x, y = SpatialCoordinate(mesh)
init_expr = sin((2*pi*x/Lx) + (2*pi*y/Ly)) + sin(2*pi*x/Lx) + sin(2*pi*y/Ly)

u_init = Function(V)
u_init.interpolate(init_expr)
unp1.assign(u_init)


eta = Constant(float(config.get("c0ip_penalty", 5.0)))

def a2d(u, v):
    n = FacetNormal(mesh)
    h = avg(CellVolume(mesh))/FacetArea(mesh)
    eqn = div(grad(u)) * div(grad(v)) * dx
    eqn += avg(div(grad(u))) * jump(grad(v),n) * dS
    eqn += avg(div(grad(v))) * jump(grad(u),n) * dS
    eqn += eta/h * jump(grad(v),n) * jump(grad(u),n) * dS
    return eqn

F = (
    v*(unp1 - un)*dx
    - dt*alpha*inner(grad(uh),grad(v))*dx
    + a2d(dt*beta*uh, v)
    - dt*gamma*0.5*(dot(grad(uh), grad(uh))- uh_mean)*v*dx
    )

solver_params = {
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
    solver_params["snes_monitor"] = None

KSProb = NonlinearVariationalProblem(F, unp1)
KSSolver = NonlinearVariationalSolver(KSProb,
                                      solver_parameters=solver_params)



un.assign(unp1)


uout = Function(Vdg)
uout.interpolate(unp1)
outfile.write(uout)



t = 0
while t < T:
    t += dt
    if config.get("verbose", False):
        PETSc.Sys.Print(f"t={t}")
    KSSolver.solve()
    un.assign(unp1)
    if dumpt > tdump - dt/2:
        uout.interpolate(unp1)
        outfile.write(uout)
        
        dumpt -= tdump
    dumpt += dt

