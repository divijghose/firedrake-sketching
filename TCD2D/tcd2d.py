# TCD2D/tcd2d.py
"""Firedrake solver for the 2D time-dependent convection-diffusion problem."""

from firedrake import *
import os

os.makedirs("results", exist_ok=True)
output_dir = "cd2d_output"
output_dir = f"results/{output_dir}"
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
outfile = VTKFile(f"{output_dir}/cd2d.pvd")

N = 512
# mesh = UnitSquareMesh(N, N)
mesh = SolidTorusMesh(1.0, 0.35, N)
V = FunctionSpace(mesh, "CG", 2)

V_vel = VectorFunctionSpace(mesh, "CG", 1)
diffusivity = Constant(0.01)

x, y, z= SpatialCoordinate(mesh)
# radial velocity field
velocity_expr = as_vector((2.0*(0.5-y), 2.0*(x - 0.5), 0.0))
vel = Function(V_vel).interpolate(velocity_expr)
f = Function(V).interpolate(Constant(0.0))

u = TrialFunction(V)
v = TestFunction(V)
u_old = Function(V)
u_new = Function(V)
u_init = Function(V)
f = Function(V)
f.interpolate(Constant(0.0))
t = 0.0
dt = 0.01
t_max = 10.0

init_expr = exp(-5000.0 * ((x-0.65)**2 + (y-0.5)**2))

u_init.interpolate(init_expr)
u_old.assign(u_init)
u_new.assign(u_init)
outfile.write(u_new, time=float(t))

bcs = [DirichletBC(V, Constant(0.0), "on_boundary")]
a = inner(u, v)*dx + dt*diffusivity*inner(grad(u), grad(v))*dx + dt*inner(dot(vel, grad(u)), v)*dx
L = inner(u_old, v)*dx + dt*inner(f, v)*dx

LinearProblem = LinearVariationalProblem(a, L, u_new, bcs=bcs)
CDSolver = LinearVariationalSolver(LinearProblem)
while t < t_max:
    u_old.assign(u_new)
    CDSolver.solve()
    t += dt
    outfile.write(u_new, time=float(t))





