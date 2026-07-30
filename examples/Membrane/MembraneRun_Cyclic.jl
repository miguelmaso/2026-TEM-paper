using HyperFEM
using JLD2
using Plots
using Plots.PlotMeasures

include("Membrane.jl")

## Problem data

problem_data = (
  width = 0.05,     # 5 cm (frame dimensions)
  thick0 = 0.001,   # 1.0 mm (undeformed)
  voltage = 5000,   # V
  freq = 10,        # Hz
  prestretch = 3.0, # -
  θr = 293.15,      # K
  t_end = 10.0,     # s
  Δt = 0.001,       # s
  ndivisions = 10,  # -
  order = 2         # -
)


## Boundary conditions

dir_u_tags = ["faces", "center_axis", "x_sym", "y_sym"]
dir_u_values = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
dir_u_time = [_->1, _->1, _->1, _->1]
dirichlet_u_masks = [[true,true,true], [true,true,false], [true,false,false], [false,true,false]]
dirichlet_u = DirichletBC(dir_u_tags, dir_u_values, dir_u_time)

dir_φ_tags = ["top_electrode", "bottom_electrode"]
dir_φ_values = [problem_data.voltage, 0.0]
dir_φ_time = [t->sin(2π*problem_data.freq*t), t->1]
dirichlet_φ = DirichletBC(dir_φ_tags, dir_φ_values, dir_φ_time)

dirichlet_θ = NothingBC()

bc = (;
  dirichlet_u_masks,
  dirichlet_u,
  dirichlet_φ,
  dirichlet_θ
)
problem_data = merge(problem_data, bc)


## Run the problem

solve_problem(problem_data)
