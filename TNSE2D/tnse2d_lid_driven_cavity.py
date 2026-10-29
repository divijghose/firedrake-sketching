from firedrake import *
from firedrake.petsc import PETSc

N = 64

mesh = UnitSquareMesh(N, N)
dt = 1e-3
T = 5e-1

def du_dt(u, u_n, dt):
    return (u - u_n) / dt

V = VectorFunctionSpace(mesh, "CG", 2)
W = FunctionSpace(mesh, "CG", 1)

Z = V * W

up = TrialFunctions(Z)
u, p = split(up)

up_n = Function(Z)
u_n, p_n = split(up_n)

up_np1 = Function(Z)
u_np1, p_np1 = split(up_np1)

v, phi = TestFunctions(Z)


bcs = [DirichletBC(Z.sub(0),Constant((1.0,0.0)),(4,)), DirichletBC(Z.sub(0),Constant((0.0,0.0)),(1,2,3))]
Re = Constant(100.0)
F = (
    inner(du_dt(u, u_n, dt),v)*dx
    + 1/Re*inner(grad(u), grad(v))*dx
    + inner(dot(grad(u), u), v)*dx 
    - p * div(v)*dx +
    div(u)*phi*dx
)

nullspace = MixedVectorSpaceBasis(
    Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

t = 0.0
outfile = VTKFile("lid_driven_cavity.pvd")
while t < T:
    up_n.assign(up_np1)
    solve(F == 0, up_np1, bcs=bcs, nullspace=nullspace,
                    solver_parameters={"snes_monitor": None,
                             "ksp_type": "gmres",
                             "mat_type": "aij",
                             "pc_type": "lu",
                             "pc_factor_mat_solver_type": "mumps"})
    t += dt
    u_np1, p_np1 = up_np1.subfunctions
    u_np1.rename("velocity")
    p_np1.rename("pressure")
    outfile.write(u_np1, p_np1, time=t)

