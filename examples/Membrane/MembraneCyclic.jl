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

m, uh = solve_problem(problem_data)

## Metrics visualization and check

η_ref = m.ηtot[1]
p1 = plot(m.time, m.ηtot, labels="Entropy", style=:solid, lcolor=:black, width=2, ylim=[1-5.1e-3, 1+5.1e-3]*η_ref, yticks=[1-5e-3, 1, 1+5e-3]*η_ref, margin=8mm, xlabel="Time [s]", ylabel="Entropy [J/K]")
p1 = plot!(p1, m.time, NaN.*m.time, labels="Temperature", style=:dash, lcolor=:gray, width=2)
p1 = plot!(twinx(p1), m.time, m.θavg, labels="Temperature", style=:dash, lcolor=:gray, width=2, xticks=false, legend=false, ylabel="Temperature [ºK]")
Ψint = m.Ψmec + m.Ψele + m.Ψthe
Ψtot = Ψint - m.Ψdir
p2 = plot(m.time, [Ψint m.Ψdir m.Dvis], labels=["̇Ψu+Ψφ+Ψθ" "Ψφ,Dir" "Dvis"], style=[:solid :dash :dashdot], lcolor=[:black :black :gray], width=2, margin=8mm, xlabel="Time [s]", ylabel="Power [W]")
p3 = plot(m.λ, m.V ./1000, labels="λp=$(problem_data.prestretch)", color=:black, width=2, margin=8mm, xlabel="λ [-]", ylabel="Voltage [kV]")
p4 = plot(p1, p2, p3, layout=@layout([a b c]), size=(1500, 500))
display(p4);


trapz(a::AbstractArray) = sum(a) -0.5(a[1] + a[end])

Dvis_θ = m.Dvis ./ m.θavg
Dvis_int = trapz(Dvis_θ) * problem_data.Δt
@show m.ηtot[end] - m.ηtot[1]
@show m.ηtot[end] - m.ηtot[1] - Dvis_int

@show trapz(Dvis_θ ./ m.cv)
@show trapz(m.∂Pθ_F ./ m.cv)
@show trapz(m.∂Dθ_E ./ m.cv)

plot(m.time, m.θavg .- 273.15, lw=2, lcolor=:black, label=nothing, xlabel="Time [s]", ylabel="Temperature [ºC]", size=(1200,400), margins=30px)
