from firedrake import *
from firedrake.petsc import PETSc

N = 64

mesh = Mesh("assets/meshes/flow_past_cylinder.msh")

dt = 1e-3
T = 5e-1
t = 0.0

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



# bcs = [DirichletBC(Z.sub(0),Constant((1.0,0.0)),(4,)), DirichletBC(Z.sub(0),Constant((0.0,0.0)),(1,2,3))]
# bcs = [DirichletBC(Z.sub(0),Constant((1.0,0.0)),(4,)), DirichletBC(Z.sub(0),Constant((0.0,0.0)),(1,2,3))]
c = Constant(t)

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

nullspace = MixedVectorSpaceBasis(
    Z, [Z.sub(0), VectorSpaceBasis(constant=True)])


outfile = VTKFile("lid_driven_cavity.pvd")
while t < T:
    up.assign(up_n)
    solve(F == 0, up, bcs=bcs, nullspace=nullspace,
                    solver_parameters={"snes_monitor": None,
                             "ksp_type": "gmres",
                             "mat_type": "aij",
                             "pc_type": "lu",
                             "pc_factor_mat_solver_type": "mumps"})
    up_n.assign(up)
    t += dt
    u, p = up.subfunctions
    u.rename("velocity")
    p.rename("pressure")
    outfile.write(u, p, time=t)

