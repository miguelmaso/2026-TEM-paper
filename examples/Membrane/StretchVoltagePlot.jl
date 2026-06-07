using JLD2
using Plots
using LaTeXStrings

default(
    fontfamily     = "Computer Modern",
    legendfontsize = 12,
    tickfontsize   = 11,
    guidefontsize  = 14,
    titlefontsize  = 12,
    palette        = :seaborn_colorblind,
    linewidth      = 3
)

function load_λV(path)
    @load path metrics
    return metrics.λ, metrics.V
end

res_path = abspath(dirname(@__FILE__), "results/")
fig_path = abspath(dirname(@__FILE__), "../../article/figures/membrane/")

p=plot(xlabel="Stretch [-]", ylabel="Voltage [kV]")
for λp in [1.5, 2.0, 2.5, 3.0]
    λ, V = load_λV("$(res_path)/Membrane_metrics_$(λp).jld2")
    plot!(λ, V ./1000, label=L"\lambda_p="*string(λp))
end
display(p);
savefig(p, joinpath(fig_path, "stretch_voltage.pdf"))
