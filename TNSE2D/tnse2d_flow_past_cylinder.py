from firedrake import *
from firedrake.petsc import PETSc
import os

mesh = Mesh("assets/meshes/flow_past_cylinder.msh")
output_dir = "tnse2d_output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

dt = 1e-3
T = 20.0
t = 0.0
c = Constant(t)

def du_dt(u, u_n, dt):
    return (u - u_n) / dt

def inlet_velocity(y, t):
    U = 1.5*sin(pi*t/8)
    inlet_profile = 4*U*y*(0.41-y)/0.41**2
    return inlet_profile

x, y = SpatialCoordinate(mesh)


V = VectorFunctionSpace(mesh, "CG", 2)
W = FunctionSpace(mesh, "CG", 1)

Z = V * W

up = Function(Z)
u, p = split(up)

up_n = Function(Z)
u_n, p_n = split(up_n)

v, phi = TestFunctions(Z)

boundary_markers = {"Bottom Wall": 1,
                    "Outlet": 2,
                    "Top Wall": 3,
                    "Inlet": 4,
                    "Cylinder": 5}

no_slip_bc = Constant((0.0, 0.0))
bcs = [DirichletBC(Z.sub(0), no_slip_bc, (boundary_markers["Bottom Wall"], boundary_markers["Top Wall"], boundary_markers["Cylinder"]))]

bcs.append(DirichletBC(Z.sub(0),(inlet_velocity(y, c),0.0), (boundary_markers["Inlet"],)))
bcs.append(DirichletBC(Z.sub(1), Constant(0.0), (boundary_markers["Outlet"],)))

Re = Constant(100.0)
F = (
    inner(du_dt(u, u_n, dt),v)*dx
    + 1/Re*inner(grad(u), grad(v))*dx
    + inner(dot(grad(u), u), v)*dx 
    - p * div(v)*dx +
    div(u)*phi*dx
)

solver_parameters = {
    "snes_type": "newtonls",
    "snes_monitor": None,
    "snes_rtol": 1e-6,
    "ksp_type": "gmres",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "full",
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_pc_type": "hypre",
    "fieldsplit_1_ksp_type": "preonly",
    "fieldsplit_1_pc_type": "hypre"
}
problem = NonlinearVariationalProblem(F, up, bcs=bcs)
solver = NonlinearVariationalSolver(problem, solver_parameters=solver_parameters)

tdump = 0.1
dumpt = 0.
outfile = VTKFile(f"{output_dir}/flow_past_cylinder.pvd")
while t < T:
    t += dt
    c.assign(t)
    PETSc.Sys.Print(f"t = {t}")
    up.assign(up_n)
    solver.solve()

    up_n.assign(up)

    if dumpt > tdump - dt/2:
        u, p = up.subfunctions
        u.rename("velocity")
        p.rename("pressure")
        outfile.write(u, p, time=t)
        
        dumpt -= tdump
    dumpt += dt
    

