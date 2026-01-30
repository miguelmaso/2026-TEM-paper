#------------------------------------------
#------------------------------------------
#
# Calibration of VHB 4905 polymer.
# Data and libraries from:
# https://doi.org/10.1016/j.ijnonlinmec.2019.103263 - Liao, Mokarram et al., 2022, On thermo-viscoelastic experimental characterization and numerical modelling of VHB polymer
# https://doi.org/10.1002/zamm.201400110 - Dippel et al., 2015, Thermo-mechanical couplings in elastomers - experiments and modelling
# https://docs.sciml.ai/Optimization/stable/#Citation - Kumar, 2023, Optimization.jl: A unified optimization package
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

μe::Float64  = 1.0e4
μ1::Float64  = 4.0e4
p1::Float64  = 0.0
cv0::Float64 = 1000.0
α::Float64   = 1.0
γv::Float64  = 0.5
γd::Float64  = 0.5

# 1283.88
# 0.7777

heating_data = read_data(joinpath(@__DIR__, "Dippel 2015.csv"), HeatingTest)
mechanical_data = read_data(joinpath(@__DIR__, "Liao_Mokarram 2022.csv"), LoadingTest)

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
function thermal_characterization(data)
  p0 = [1.0e3, 0.5]  # Initial seed
  lb = [ 10.0, 0.0]  # Minimum search limits
  ub = [1.0e5, 1.0]  # Maximum search limits
  opt_func = OptimizationFunction(loss)   # AutoFiniteDiff() is needed to be provided for gradient-based search algorithms
  opt_prob = OptimizationProblem(opt_func, p0, data, lb=lb, ub=ub)
  solve(opt_prob, ECA(), maxiters=10000, maxtime=60.0)  # ECA (Evolutionary Centers Algorithm), NelderMead, LBFGS
end

function plot_experiment(model, data::HeatingTest, p=plot())
  cv_values = simulate_experiment(model, data.θ)
  plot!(p, data.θ.-K0, [cv_values, data.cv], label=["Model" "Experiment"], xlabel="T [ºC]", ylabel="cv [J/(kg·ºK)]", lw=2, mark=[:none :circle], markerstrokewidth=0)
end

sol_heat = thermal_characterization(heating_data)
cv0, γv = sol_heat.u
R2_heat = 1-sol_heat.objective
println("Optimum cv0 : ", cv0)
println("Optimum γv :  ", γv)
println("R2 :          ", 100R2_heat)

# Plot the solution
model = build_constitutive_model(cv0, γv)
pl1 = plot_experiment(model, heating_data[1])
display(pl1)

#------------------------------------------
# Visco-elastic characterization
#------------------------------------------
function mechanical_characterization(data)
  p0 = [  1e4,   4.0e5,      0.0,  500.0, 0.5]  # Initial seed
  lb = [100.0,   100.0,     -5.0,   10.0, 0.0]  # Minimum search limits
  ub = [5.0e5,   2.0e5,      5.0, 1000.0, 1.0]  # Maximum search limits
  opt_func = OptimizationFunction(loss)   # AutoFiniteDiff() is needed for gradient-based search algorithms
  opt_prob = OptimizationProblem(opt_func, p0, data, lb=lb, ub=ub)
  solve(opt_prob, ECA(), maxiters=100000, maxtime=60.0)  # ECA (Evolutionary Centers Algorithm), NelderMead, LBFGS
end

function plot_experiment(model, data::LoadingTest, p=plot())
  σ_values = simulate_experiment(model, data.θ, data.Δt, data.λ)
  plot!(p, data.λ, [σ_values, data.σ], label=["Model" "Experiment"], mark=[:none :circle], lw=2, markerstrokewidth=0)
end

sol_mech = mechanical_characterization(mechanical_data)
μe, μ1, p1, α, γd = sol_mech.u
R2_mech = 1-sol_mech.objective
println("Optimum μe : ", μe)
println("Optimum μ1 : ", μ1)
println("Optimum τ1 : ", exp(p1))
println("Optimum α :  ", α)
println("Optimum γd : ", γd)
println("R2 :         ", 100R2_mech)

model = build_constitutive_model(sol_mech.u...)
pl2 = plot_experiment(model, mechanical_data[12])
display(pl2);

# cons_model = build_constitutive_model(1.37e4, 5.64e4, log(0.82), 1280.0, 100.0, 0.77, 0.5)
# plot_experiment(cons_model, mechanical_data[1])
