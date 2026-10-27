""" This code was taken from Ed Bueler's lectures on Variational
Inequalities"""
from firedrake import *
import os

os.makedirs("results/vi", exist_ok=True)
N = 32
mesh = RectangleMesh(N, N, 2, 2, -2, -2, diagonal="crossed")
x, y = SpatialCoordinate(mesh)
r2 = x**2 + y**2
psi = conditional(r2<1, sqrt(1-r2), 0)

V = FunctionSpace(mesh, "CG", 1)
psih = Function(V, name="psi_h").interpolate(psi)
uh = Function(V, name="u_h")
vh = TestFunction(V)

F = inner(grad(uh), grad(vh))*dx

bcs = DirichletBC(V, 0, "on_boundary")
problem = NonlinearVariationalProblem(F, uh, bcs=bcs)

sp = {"snes_type" : "vinewtonrsls",
"snes_converged_reason" : None,}

solver = NonlinearVariationalSolver(problem, solver_parameters=sp)

INF = Function(V).interpolate(Constant(PETSc.INFINITY))
solver.solve(bounds=(psih, INF))
VTKFile(f"results/vi/obstacle_problem.pvd").write(uh, psih)