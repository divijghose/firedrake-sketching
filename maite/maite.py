from firedrake import *


L = 2.0 * pi
N = 256
mesh = PeriodicIntervalMesh(N, length=L)
V = FunctionSpace(mesh, "CG", 1)
W = V * V


dt_value = 1.0e-4
T = 5.0
dt = Constant(dt_value)
num_steps = int(round(T / dt_value))

U = Function(W, name="solution")
eta, w = split(U)

U_old = Function(W, name="solution_old")
eta_old, w_old = split(U_old)

v, chi = TestFunctions(W)

x = SpatialCoordinate(mesh)
eta0 = 0.5 * cos(2.0 * pi * x[0] / L)

U_old.sub(0).interpolate(eta0)
U_old.sub(1).interpolate(eta0)
U.assign(U_old)


f = eta**2 + eta**3 + eta * (1.0 - eta)**2
q = (1.0 - eta)**3

dxq = dx(metadata={"quadrature_degree": 6})

F = (
    ((eta - eta_old) / dt) * v * dxq
    - f * v.dx(0) * dxq
    - q * w.dx(0) * v.dx(0) * dxq
    + (w - eta) * chi * dxq
    + eta.dx(0) * chi.dx(0) * dxq
)

problem = NonlinearVariationalProblem(F, U)
solver_parameters = {
    "snes_type": "newtonls",
    "snes_rtol": 1.0e-8,
    "snes_atol": 1.0e-10,
    "snes_max_it": 20,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}
solver = NonlinearVariationalSolver(
    problem,
    solver_parameters=solver_parameters,
)

outfile = VTKFile("eta.pvd")
for n in range(num_steps):
    solver.solve()

    U_old.assign(U)

    if (n + 1) % 100 == 0:
        print(f"step={n + 1}, time={(n + 1) * dt_value:.6e}")
        eta_final = U.sub(0)

        outfile.write(eta_final, time=(n + 1) * dt_value)


