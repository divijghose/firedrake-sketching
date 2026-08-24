
from firedrake import *
from math import pi
import numpy as np
import matplotlib.pyplot as plt
import os 

output_dir = "ks_output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)



N = 128
v1 = 0.035
v2 = 0.035
Lx = np.sqrt(np.pi**2/v1)
Ly = np.sqrt(np.pi**2/v2)
mesh = PeriodicRectangleMesh(nx=N, ny=N, Lx=2*Lx, Ly=2*Ly)
dt = 1e-1
T = 100
alpha = Constant(1.0) # viscosity
beta = Constant(1.0) # hyperviscosity
gamma= Constant(1.0) # advection
theta = 0.5

V = FunctionSpace(mesh, "CG", 2)
Vdg = FunctionSpace(mesh, "CG", 1)
unp1 = Function(V)
un = Function(V)
v = TestFunction(V)

uh = (1 - theta) * un + theta * unp1
uh_mean = assemble(uh*uh*dx)/(4*Lx*Ly)

x, y = SpatialCoordinate(mesh)
# init_expr = cos(12*pi*x/Lx)*cos(12*pi*y/Ly)
init_expr = sin((2*pi*x/Lx) + (2*pi*y/Ly)) + sin(2*pi*x/Lx) + sin(2*pi*y/Ly)

u_init = Function(V)
u_init.interpolate(init_expr)
unp1.assign(u_init)


eta = Constant(5.)

def a(u, v):
    h = avg(CellVolume(mesh))/FacetArea(mesh)
    eqn = v.dx(0).dx(0)*u.dx(0).dx(0)*dx
    eqn += avg(u.dx(0).dx(0))*jump(v.dx(0))*dS
    eqn += avg(v.dx(0).dx(0))*jump(u.dx(0))*dS
    eqn += eta/h*jump(v.dx(0))*jump(u.dx(0))*dS
    return eqn
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

params = {
    "snes_type": "newtonls",
    "snes_rtol": 1e-50,
    "snes_atol": 1e-50,
    "snes_stol": np.sqrt(np.finfo(np.float64).eps) * 1e-2,
    "ksp_type": "preonly",
    "snes_linesearch_type": "none",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "snes_monitor": None,
}

#make the solver
KSProb = NonlinearVariationalProblem(F, unp1)
KSSolver = NonlinearVariationalSolver(KSProb,
                                      solver_parameters=params)



un.assign(unp1)
tdump = 1e-2
dumpt = 0.

file = VTKFile(f"{output_dir}/1_tks_2d.pvd")
uout = Function(Vdg)
uout.interpolate(unp1)
file.write(uout)



t = 0
while t < T:
    t += dt
    KSSolver.solve()
    un.assign(unp1)

    if dumpt > tdump - dt/2:
        uout.interpolate(unp1)
        file.write(uout)
        
        dumpt -= tdump
    dumpt += dt

