using Plots

include("ConstitutiveModelling.jl")
include("ExperimentsData.jl")

data = read_data(joinpath(@__DIR__, "Liao_Mokarram 2022.csv"), LoadingTest)

g(θ, θr, γ) = 1 + 1/(γ+1) * ((θ/θr)^(-1(γ+1)) - 1)

θ_values = 253:10:333
g_values = g.(θ_values, 293, 45.1)
plot(θ_values .-273, g_values)

ex = data[1:3:end]

θ_exp = [e.θ for e in ex]
μ_avg = [e.σ_max / e.λ_max for e in ex]
plot(θ_exp .- 273, μ_avg ./ μ_avg[3], xlabel="T [ºC]", ylabel="μ/μR [KPa]")
