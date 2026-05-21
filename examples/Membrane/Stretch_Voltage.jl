using Plots
using JLD2

default(palette=:seaborn_colorblind)

function load_var(filepath, field)
  @load filepath m
  getfield(m, field)
end

@load abspath(dirname(@__FILE__), "results/Membrane_metrics_1.5_10000.jld2") m
λ15 = m.λ
V15 = m.V

@load abspath(dirname(@__FILE__), "results/Membrane_metrics_2.0_9000.jld2") m
λ20 = m.λ
V20 = m.V

@load abspath(dirname(@__FILE__), "results/Membrane_metrics_3.0_6000.jld2") m
λ30 = m.λ
V30 = m.V


plot([λ15, λ20, λ30], [V15, V20, V30], labels=["λp=1.5" "λp=2.0" "λp=3.0"], xlabel="λ [-]", ylabel="Voltage [kV]", lw=3)
