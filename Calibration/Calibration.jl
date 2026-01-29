#------------------------------------------
#------------------------------------------
#
# Calibration of VHB 4905 polymer
# Liao, Mokarram et al., 2022, On thermo-viscoelastic experimental characterization and numerical modelling of VHB polymer
# Dippel et al., 2015, Thermo-mechanical couplings in elastomers - experiments and modelling
#
#------------------------------------------
#------------------------------------------

using Plots
using Optimization, OptimizationOptimJL, ForwardDiff, OptimizationMetaheuristics

include("ConstitutiveModelling.jl")
include("ExperimentsData.jl")

#------------------------------------------
# Objective function
#------------------------------------------

function loss(model::PhysicalModel, data::Vector{T} where T<:ExperimentData)
  error = 0
  for exp_data ∈ data
    σ_model = simulate_experiment(model, exp_data.θ, exp_data.Δt, exp_data.λ)
    σ_err = (σ_model .- exp_data.σ) / exp_data.σ_max
    error += sqrt(sum(abs2, σ_err) / length(σ_err)) * exp_data.weight
  end
  error / length(data)
end

function global_loss(params, data)
  model = build_constitutive_model(params...)
  err = loss(model, data)
end

function build_constitutive_model(μ, μ1, p1, α, γv, γd)
  long_term = NeoHookean3D(λ=100μ, μ=μ)
  branch_1 = ViscousIncompressible(IncompressibleNeoHookean3D(λ=0., μ=μ1), τ=exp(p1))
  visco_elasto = GeneralizedMaxwell(long_term, branch_1)
  thermal_model = ThermalModel3rdLaw(cv0=10.0, θr=293.15, α=α, κ=1.0, γv=γv, γd=γd)
  cons_model = ThermoMechModel(thermal_model, visco_elasto)
end

#------------------------------------------
# Visco-elastic characterization
#------------------------------------------

experiments_data = read_data(joinpath(@__DIR__, "Liao_Mokarram 2022.csv"), LoadingTest)

p0 = [1000.0, 1000.0, log(0.2),  1.0, 0.5, 0.5]  # Initial seed
lb = [100.0,   100.0,     -5.0,  0.0, 0.0, 0.0]  # Minimum search limits
ub = [5.0e4,  10.0e4,      5.0, 10.0, 1.0, 1.0]  # Maximum search limits

opt_func = OptimizationFunction(global_loss)   # AutoFiniteDiff() is needed for gradient-based search algorithms
opt_prob = OptimizationProblem(opt_func, p0, experiments_data, lb=lb, ub=ub)

sol = solve(opt_prob, ECA(), maxiters=100000, maxtime=300.0)  # ECA (Evolutionary Centers Algorithm), NelderMead, LBFGS
println("p optimo: ", sol.u)
println("Error final: ", sol.minimum)



# long_term = NeoHookean3D(λ=1.0e6, μ=1.37e4)
# branch_1 = ViscousIncompressible(IncompressibleNeoHookean3D(λ=0., μ=5.64e4), τ=0.82)
# visco_elasto = GeneralizedMaxwell(long_term, branch_1)
# thermal_model = ThermalModel3rdLaw(cv0=10.0, θr=293.15, α=1.0, κ=1.0, γv=0.5, γd=0.5)
# cons_model = ThermoMechModel(thermal_model, visco_elasto)

# e1 = experiments_data[1]
# σ1 = simulate_experiment(cons_model, 293.15, e1.Δt, e1.λ)
# plot(e1.λ, σ1)