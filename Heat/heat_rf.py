from firedrake import *
from firedrake.adjoint import *
from pyadjoint import *
from firedrake import exp
import os

os.makedirs("results", exist_ok=True)
output_dir = "heat_rf_output"
output_dir = f"results/{output_dir}"
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
outfile = VTKFile(f"{output_dir}/heat_rf.pvd")

N = 64
mesh = UnitSquareMesh(N, N)
V = FunctionSpace(mesh, "CG", 2)
continue_annotation()

k = Constant(0.1)

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

x, y = SpatialCoordinate(mesh)
init_expr = exp(-200.0 * ((x-0.5)**2 + (y-0.5)**2))
u_init.interpolate(init_expr)
u_old.assign(u_init)
u_new.assign(u_init)
pause_annotation()
outfile.write(u_new, time=float(t))
continue_annotation()
bcs = [DirichletBC(V, Constant(0.0), "on_boundary")]
a = inner(u, v)*dx + dt*k*inner(grad(u), grad(v))*dx
L = inner(u_old, v)*dx + dt*inner(f, v)*dx

LinearProblem = LinearVariationalProblem(a, L, u_new, bcs=bcs)
HeatSolver = LinearVariationalSolver(LinearProblem)
while t < 4*dt:
    u_old.assign(u_new)
    HeatSolver.solve()
    t += dt
pause_annotation()
print("Comes here")
outfile.write(u_new, time=float(t))
tape = get_working_tape()
tape.visualise("heat_rf_tape.pdf")

Jhat = ReducedFunctional(u_new, controls=Control(k), parameters = u_init)

while t < t_max:
    u_new.assign(Jhat(k))
    t += 4*dt
    Jhat.update_parameters(u_new)
    outfile.write(u_new, time=float(t))




