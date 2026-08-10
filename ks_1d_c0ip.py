from firedrake import *
from math import pi
import numpy as np
ncells = 1000
L = 32*pi
mesh = PeriodicIntervalMesh(ncells, L)

V = FunctionSpace(mesh, "CG", 2)
Vdg = FunctionSpace(mesh, "DG", 1)
un = Function(V)
unp1 = Function(V)

uh = (un + unp1)/2

v = TestFunction(V)

dt = 0.01
dT = Constant(dt)

alpha = Constant(1.0) # viscosity
beta = Constant(0.02923) # hyperviscosity
gamma = Constant(1.) # advection

eta = Constant(5.)

def a(u, v):
    h = avg(CellVolume(mesh))/FacetArea(mesh)
    eqn = v.dx(0).dx(0)*u.dx(0).dx(0)*dx
    eqn += avg(u.dx(0).dx(0))*jump(v.dx(0))*dS
    eqn += avg(v.dx(0).dx(0))*jump(u.dx(0))*dS
    eqn += eta/h*jump(v.dx(0))*jump(u.dx(0))*dS
    return eqn

eqn = (
    v*(unp1 - un)*dx
    - dT*alpha*v.dx(0)*uh.dx(0)*dx
    + a(dT*beta*uh, v)
    - dT*gamma*0.5*v.dx(0)*uh*uh*dx
    )

params = {
    "snes_atol": 1.0e-50,
    "snes_rtol": 1.0e-6,
    "snes_stol": 1.0e-50,
    "ksp_type":"preonly",
    "pc_type":"lu"
}

#make the solver
KSProb = NonlinearVariationalProblem(eqn, unp1)
KSSolver = NonlinearVariationalSolver(KSProb,
                                      solver_parameters=params)

#initial condition

x, = SpatialCoordinate(mesh)
# un.project(exp(sin(pi*2*x) + 0.2*cos(pi*x)))
un.project(cos(x / 16.0) * (1.0 + sin(x / 16.0)))

t = 0.
tmax = 2
tdump = 0.1
dumpt = 0.

file0 = VTKFile("stuff.pvd")
uout = Function(Vdg)
uout.interpolate(un)
file0.write(uout)

Vplot = FunctionSpace(mesh, "CG", 1)
uplot = Function(Vplot)
uplot.interpolate(un)

u_data = []
t_data = []
u_data.append(uplot.dat.global_data[:])
t_data.append(0.0)
while t < tmax - dt/2:
    t += dt

    KSSolver.solve()
    print(norm(unp1))
    un.assign(unp1)

    uplot.interpolate(un)
    u_data.append(uplot.dat.global_data[:])
    t_data.append(t)
    if dumpt > tdump - dt/2:
        uout.interpolate(un)
        file0.write(uout)
        
        dumpt -= tdump
    dumpt += dt

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
X, Tt = np.meshgrid(np.linspace(0, L, ncells), t_data)
ax.contourf(X, Tt, u_data)
plt.xlabel("x")
plt.ylabel("t")
plt.title("1D Kuramoto-Sivashinsky")
plt.savefig("ks_1d.png", dpi=300, bbox_inches="tight")