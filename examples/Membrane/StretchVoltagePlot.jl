using CSV
using JLD2
using Plots
using DataFrames
using LaTeXStrings

default(
    fontfamily     = "Computer Modern",
    legendfontsize = 11,
    tickfontsize   = 11,
    guidefontsize  = 14,
    titlefontsize  = 12,
    palette        = :seaborn_colorblind,
    linewidth      = 3,
    mswidth        = 0
)

function load_λV(path)
    @load path metrics
    return metrics.λ, metrics.V
end

src_path = abspath(dirname(@__FILE__), "data/")
res_path = abspath(dirname(@__FILE__), "results/")
fig_path = abspath(dirname(@__FILE__), "../../article/figures/membrane/")


## Raw simulation

p=plot(xlabel="Stretch [-]", ylabel="Voltage [kV]")
for λp in [1.5, 2.0, 2.5, 3.0]
    λ, V = load_λV("$(res_path)/Membrane_metrics_$(λp).jld2")
    plot!(λ, V ./1000, label=L"\lambda_p="*string(λp))
end
display(p);
savefig(p, joinpath(fig_path, "stretch_voltage.pdf"))


## Comparison

df = CSV.read("$(src_path)/benchmark.csv", DataFrame; decimal=',')

markers = Dict(
  "Godaba 2017" => :circle,
  "Benham 2025" => :utriangle
)
colors = Dict(
  2.0 => 1,
  3.0 => 2,
  4.0 => 3
)

p=plot(xlabel="Stretch [-]", ylabel="Voltage [kV]")
for λp in [2.0, 3.0, 4.0]
  λ, V = load_λV("$(res_path)/Membrane_metrics_$(λp).jld2")
  plot!(λ, V ./1000, label=false, color=colors[λp])

  λ_max = λ[end]
  V_max = V[end] / 1000

  benchmarks = subset(df, :prestretch => p -> p .== λp)
  for experim in groupby(benchmarks, :author)
    author = experim.author[1]
    marker = markers[author]
    color = colors[λp]
    λ_last_i = findlast(λ -> λ < λ_max, experim.stretch)
    V_last_i = findlast(V -> V < V_max, experim.voltage)
    last_i = max(λ_last_i, V_last_i)
    # if author == "Benham 2025" && λp == 4.0
    #   experim.stretch .+= 0.25
    #   experim.voltage .*= 0.95
    # end
    scatter!(experim.stretch[1:last_i], experim.voltage[1:last_i], marker=marker, color=color, label=false)
  end
end

plot!([], label=L"\lambda_p=2.0", color=colors[2.0])
plot!([], label=L"\lambda_p=3.0", color=colors[3.0])
plot!([], label=L"\lambda_p=4.0", color=colors[4.0])
scatter!([], label="Godaba 2017", color=:black, marker=markers[:"Godaba 2017"])
scatter!([], label="Benham 2025", color=:black, marker=markers[:"Benham 2025"])
plot!([], label="simulation", color=:black)

display(p);
savefig(p, joinpath(fig_path, "stretch_voltage_comparison.pdf"))
