from firedrake import *
import numpy as np
import matplotlib.pyplot as plt
import os

output_dir = "ch_output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

N = 64
L = 1.0
mesh = PeriodicUnitSquareMesh(N, N)

V = FunctionSpace(mesh, "CG", 1)    