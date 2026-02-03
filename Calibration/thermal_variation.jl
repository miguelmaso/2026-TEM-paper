using Plots

include("ConstitutiveModelling.jl")
include("ExperimentsData.jl")

data = read_data(joinpath(@__DIR__, "Liao_Mokarram 2022.csv"), LoadingTest)

g(θ, θr, γ) = 1 + 1/(γ+1) * ((θ/θr)^(-1(γ+1)) - 1)

θ_values = 253:10:333
g_values = g.(θ_values, 293, 45.1)
plot(θ_values .-273, g_values)

ex = data[1:3:end]
ex = [record for record in data if record.λ_max == 2]

θ_exp = [e.θ for e in ex]
μ_avg = [e.σ_max / e.λ_max for e in ex]
g_exp = μ_avg ./ μ_avg[3]
ann   = [text(@sprintf("%.2f",g),8,:left,:bottom) for g in g_exp]
p = plot(θ_exp .- 273, g_exp, series_ann=ann, label="θr = 40 ºC", xlabel="T [ºC]", ylabel="g = μ / μR [-]", mark=:circle, lw=2, msw=0)
display(p);

p = plot()
for i in 1:length(ex)
  plot!(ex[i].λ, ex[i].σ ./ g_exp[i], label=@sprintf("%3.0f ºC",ex[i].θ-K0), xlabel="Stretch [-]", ylabel="Stress / g(θ) [Pa]", mark=:circle, lw=2, msw=0)
end
display(p);
