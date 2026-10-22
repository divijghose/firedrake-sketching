from firedrake import *
from firedrake.petsc import PETSc
import os
import sys

mesh = Mesh("assets/meshes/2Dmesh.msh")
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
output_dir = config.get("output_dir", "tnse2d_output")
output_dir = f"results/{output_dir}"
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
outfile = VTKFile(f"{output_dir}/tnse2d_fpc.pvd")


dt = float(config.get("dt", 1e-3))
T = float(config.get("tmax", 500.0)) *dt
t = 0.0
c = Constant(t)
tdump = config.get("output_frequency", 100) * float(dt)
dumpt = 0.
theta = 0.5 # Crank-Nicolson scheme
def du_dt(u_np1, u_, dt):
    return (u_np1 - u_) / dt

def inlet_velocity(y):
    U = 1.5
    L = 0.41
    inlet_profile = 4 * U * y * (L - y) / L**2 
    return as_vector([inlet_profile, 0.0])

x, y = SpatialCoordinate(mesh)


V = VectorFunctionSpace(mesh, "CG", 2)
W = FunctionSpace(mesh, "CG", 1)
Z = V * W

up = Function(Z)
u, p = split(up)

up_n = Function(Z)
u_n, p_n = split(up_n)

uph = Function(Z)

uh, ph = split(uph)

uh = (theta*u + (1.0-theta)*u_n)
ph = (theta*p + (1.0-theta)*p_n)

v, phi = TestFunctions(Z)

# boundary_markers = {"Bottom Wall": 1,
#                     "Outlet": 2,
#                     "Top Wall": 3,
#                     "Inlet": 4,
#                     "Cylinder": 5}

# no_slip_bc = Constant((0.0, 0.0))
# bcs = [DirichletBC(Z.sub(0), no_slip_bc, (boundary_markers["Bottom Wall"], boundary_markers["Top Wall"], boundary_markers["Cylinder"],))]

# bcs.append(DirichletBC(Z.sub(0), interpolate(inlet_velocity(y), V), (boundary_markers["Inlet"],)))
# bcs.append(DirichletBC(Z.sub(1), Constant(0.0), (boundary_markers["Outlet"],)))

noslipBC = DirichletBC(Z.sub(0), Constant((0.0, 0.0)), (3,))
inflowBC = DirichletBC(Z.sub(0), interpolate(inlet_velocity(y),V), (1,))
cylinderBC = DirichletBC(Z.sub(0), Constant((0.0, 0.0)), (5,))

bcs = [noslipBC, inflowBC, cylinderBC]
nullspace = MixedVectorSpaceBasis(
    Z, [Z.sub(0), VectorSpaceBasis(constant=True)]
)

Re = Constant(float(config.get("reynolds_number", 100.0)))

F = (
    inner(du_dt(u, u_n, dt),v)*dx
    + Constant(1.0/Re)*inner(grad(uh), grad(v))*dx
    +inner(dot(grad(uh), uh), v)*dx 
    - ph * div(v)*dx +
    div(uh)*phi*dx
)
appctx = {"Re": Re, "velocity_space": 0}
if config.get("solver_type", "default") == "pcdpc":
    solver_parameters = {
        "mat_type": "matfree",
        "ksp_type": "fgmres",
        "ksp_gmres_modifiedgramschmidt": None,
        "ksp_monitor_true_residual": None,
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "schur",
        "pc_fieldsplit_schur_fact_type": "lower",
        "fieldsplit_0_ksp_type": "preonly",
        "fieldsplit_0_pc_type": "python",
        "fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
        "fieldsplit_0_assembled_pc_type": "lu",
        "fieldsplit_1_ksp_type": "gmres",
        "fieldsplit_1_ksp_rtol": 1e-4,
        "fieldsplit_1_pc_type": "python",
        "fieldsplit_1_pc_python_type": "firedrake.PCDPC",
        "fieldsplit_1_pcd_Mp_ksp_type": "preonly",
        "fieldsplit_1_pcd_Mp_pc_type": "lu",
        "fieldsplit_1_pcd_Kp_ksp_type": "preonly",
        "fieldsplit_1_pcd_Kp_pc_type": "lu",
        "fieldsplit_1_pcd_Fp_mat_type": "matfree",
        }
else:
    solver_parameters = {
        "snes_type": "newtonls",
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
    # solver_parameters={"snes_monitor": None,
    #                 "ksp_type": "gmres",
    #                 "mat_type": "aij",
    #                 "pc_type": "lu",
    #                 "pc_factor_mat_solver_type": "mumps"}
if config.get("verbose", False):
    solver_parameters["snes_monitor"] = None

problem = NonlinearVariationalProblem(F, up, bcs=bcs)
solver = NonlinearVariationalSolver(problem, solver_parameters=solver_parameters, nullspace=nullspace, appctx=appctx)

up_n.assign(up)


while t < T:
    t += dt
    c.assign(t)
    if config.get("verbose", False):
        PETSc.Sys.Print(f"t = {t}")
    solver.solve()

    up_n.assign(up)
    if t > 1.0*dt:
        if dumpt > tdump - dt/2:
            u, p = up.subfunctions
            u.rename("velocity")
            p.rename("pressure")
            outfile.write(u, p, time=t)
            
            dumpt -= tdump
        dumpt += dt
    

