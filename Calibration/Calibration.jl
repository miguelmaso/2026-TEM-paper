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
using HyperFEM, HyperFEM.ComputationalModels.EvolutionFunctions
using Optimization, OptimizationOptimJL, OptimizationMetaheuristics

include("ConstitutiveModelling.jl")
include("ExperimentsData.jl")

#------------------------------------------
# Objective function
#------------------------------------------

function loss(model::PhysicalModel, data::LoadingTest)
  σ_model = simulate_experiment(model, data.θ, data.Δt, data.λ)
  σ_err = (σ_model .- data.σ) / data.σ_max
  sqrt(sum(abs2, σ_err) / length(σ_err)) * data.weight
end

function loss(model::PhysicalModel, data::HeatingTest)
  cv_model = simulate_experiment(model, data.θ)
  cv_err = (cv_model .- data.cv) / data.cv_max
  sqrt(sum(abs2, cv_err) / length(cv_err)) * data.weight
end

function loss(model::PhysicalModel, data::Vector{<:ExperimentData})
  sqrt(sum(d -> loss(model, d)^2, data) / length(data))
end

function loss(params, data)
  model = build_constitutive_model(params...)
  err = loss(model, data)
end

μe  = 1.0e4
μ1  = 4.0e4
p1  = 0.0
cv0 = 1000.0
α   = 1.0
γv  = 0.5
γd  = 0.5

# 1283.88
# 0.7777

build_constitutive_model(μe, μ1, p1, α, γd) = 
  build_constitutive_model(μe, μ1, p1, cv0, α, γv, γd)

build_constitutive_model(cv0, γv) = 
  build_constitutive_model(μe, μ1, p1, cv0, α, γv, γd)

function build_constitutive_model(μe, μ1, p1, cv0, α, γv, γd)
  long_term = NeoHookean3D(λ=100μe, μ=μe)
  branch_1 = ViscousIncompressible(IncompressibleNeoHookean3D(λ=0., μ=μ1), τ=exp(p1))
  visco_elasto = GeneralizedMaxwell(long_term, branch_1)
  thermal_model = ThermalModel3rdLaw(cv0=cv0, θr=293.15, α=α, κ=1.0, γv=γv, γd=γd)
  return ThermoMechModel(thermal_model, visco_elasto)
end

#------------------------------------------
# Thermal characterization
#------------------------------------------
heating_data = read_data(joinpath(@__DIR__, "Dippel 2015.csv"), HeatingTest)

p0 = [100.0, 0.2]  # Initial seed
lb = [ 10.0, 0.0]  # Minimum search limits
ub = [10.e4, 1.0]  # Maximum search limits

opt_func = OptimizationFunction(loss)   # AutoFiniteDiff() is needed for gradient-based search algorithms
opt_prob = OptimizationProblem(opt_func, p0, heating_data, lb=lb, ub=ub)

sol = solve(opt_prob, ECA(), maxiters=10000, maxtime=60.0)  # ECA (Evolutionary Centers Algorithm), NelderMead, LBFGS
cv0, γv = sol.u...
γv  = sol.u[2]
println("Optimum cv0 : ", cv0)
println("Optimum γv :  ", γv)
println("R2 :          ", 100(1-sol.objective))

# Plot the solution
heat_1 = heating_data[1]
model = build_constitutive_model(cv0, γv)
cv_values = simulate_experiment(model, heat_1.θ)
p0 = plot(heat_1.θ.-K0, [cv_values, heat_1.cv], label=["Model" "Experiment"], xlabel="T [ºC]", ylabel="cv [J/(kg·ºK)]", lw=2, mark=[:none :circle], markerstrokewidth=0)
display(p0)

#------------------------------------------
# Visco-elastic characterization
#------------------------------------------

experiments_data = read_data(joinpath(@__DIR__, "Liao_Mokarram 2022.csv"), LoadingTest)

p0 = [1000.0, 1000.0, log(0.2),  300.0, 0.5]  # Initial seed
lb = [100.0,   100.0,     -5.0,   10.0, 0.0]  # Minimum search limits
ub = [5.0e4,  10.0e4,      5.0, 1000.0, 1.0]  # Maximum search limits

opt_func = OptimizationFunction(loss)   # AutoFiniteDiff() is needed for gradient-based search algorithms
opt_prob = OptimizationProblem(opt_func, p0, experiments_data, lb=lb, ub=ub)

sol = solve(opt_prob, ECA(), maxiters=100000, maxtime=60.0)  # ECA (Evolutionary Centers Algorithm), NelderMead, LBFGS
μe, μ1, p1, α, γd = sol.u...
println("Optimum μe : ", μe)
println("Optimum μ1 : ", μ1)
println("Optimum τ1 : ", exp(p1))
println("Optimum α :  ", α)
println("Optimum γd : ", γd)
println("R2 :         ", 100(1-sol.minimum))

exp_1 = experiments_data[1]
model = build_constitutive_model(sol.u...)
σ_values = simulate_experiment(model, exp_1.θ, exp_1.Δt, exp_1.λ)
p1 = plot(exp_1.λ, [σ_values*15, exp_1.σ], label=["Model" "Experiment"], mark=[:none :circle], lw=2, markerstrokewidth=0)


cons_model = build_constitutive_model(1.37e4, 5.64e4, log(0.82), 1280.0, 100.0, 0.77, 0.5)
e1 = experiments_data[1]
σ1 = simulate_experiment(cons_model, 293.15, e1.Δt, e1.λ)
plot(e1.λ, σ1)