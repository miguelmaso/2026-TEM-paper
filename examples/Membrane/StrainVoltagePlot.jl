using JLD2
using Plots
using LaTeXStrings

default(
    fontfamily     = "Computer Modern",
    legendfontsize = 12,
    tickfontsize   = 10,
    labelfontsize  = 14,
    titlefontsize  = 12,
    palette        = :seaborn_colorblind,
    linewidth      = 2
)

function load_λV(path)
    @load path metrics
    return metrics.λ, metrics.V
end

res_path = abspath(dirname(@__FILE__), "results/")
fig_path = abspath(dirname(@__FILE__), "../../article/figures/membrane/")

λ_15_7800, V_15_7800 = load_λV("$(res_path)/Membrane_metrics_1.5_7800.jld2")

p=plot(xlabel="λ [-]", ylabel="Voltage [kV]")
plot!(λ_15_7800, V_15_7800 ./1000, label=L"\lambda_p=1.5")

display(p);
savefig(p, joinpath(fig_path, "strain_voltage.pdf"))
