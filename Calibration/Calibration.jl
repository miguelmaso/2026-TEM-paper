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

using Plots, Printf
using HyperFEM, HyperFEM.ComputationalModels.EvolutionFunctions
using Optimization, OptimizationOptimJL, OptimizationMetaheuristics

include("ConstitutiveModelling.jl")
include("ExperimentsData.jl")

const colors2 = mapreduce(c -> [c,c], vcat, palette(:default))
const colors3 = mapreduce(c -> [c,c,c], vcat, palette(:default))

const temp_label = data -> @sprintf("%.0f", data.θ-K0) * " ºC"
const vel_label = data -> @sprintf("%.1f", data.v) * " /s"
const stretch_label = data -> "λ = " * @sprintf("%.0f", 100data.λ_max) * " %"

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
mechanical_data = [record for record in mechanical_data if record.θ > (-10+K0)]

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
println("Optimum cv0 : ", lpad(@sprintf("%.1f", cv0), 7))
println("Optimum γv :  ", lpad(@sprintf("%.2f", γv), 7))
println("R2 :          ", lpad(@sprintf("%.1f", 100R2_heat), 7))
text1 = text("cv⁰ = " * @sprintf("%.0f", cv0) * " N/(m²·K)\n" *
             "γ̄   = " * @sprintf("%.2f", γv) * "\n" *
             "R2  = " * @sprintf("%.0f", 100R2_heat) * " %",
             8, :left)

# Plot the solution
model = build_constitutive_model(cv0, γv)
pl1 = plot_experiment(model, heating_data[1])
annotate!(pl1, (0.05, 0.75), text1, relative=true)
display(pl1)

#------------------------------------------
# Visco-elastic characterization
#------------------------------------------
function mechanical_characterization(data)
  p0 = [  1e4,   4.0e5,      0.0,  500.0,  1.0]  # Initial seed
  lb = [100.0,   100.0,     -5.0,   10.0,  0.0]  # Minimum search limits
  ub = [5.0e5,   2.0e5,      5.0, 1000.0, 99.0]  # Maximum search limits
  opt_func = OptimizationFunction(loss)   # AutoFiniteDiff() is needed for gradient-based search algorithms
  opt_prob = OptimizationProblem(opt_func, p0, data, lb=lb, ub=ub)
  solve(opt_prob, ECA(), maxiters=100000, maxtime=60.0)  # ECA (Evolutionary Centers Algorithm), NelderMead, LBFGS
end

function plot_experiment(model, data::LoadingTest, labelfn=d->"", p=plot())
  σ_values = simulate_experiment(model, data.θ, data.Δt, data.λ)
  label = labelfn(data)
  plot!(p, data.λ, [σ_values, data.σ], label=[label ""], xlabel="Stretch [-]", ylabel="Stress [Pa]", typ=[:path :scatter], lw=2, mswidth=0, color_palette=colors2)
end

sol_mech = mechanical_characterization(mechanical_data)
μe, μ1, p1, α, γd = sol_mech.u
R2_mech = 1-sol_mech.objective
println("Optimum μe : ", lpad(@sprintf("%.1f", μe), 8))
println("Optimum μ1 : ", lpad(@sprintf("%.1f", μ1), 8))
println("Optimum τ1 : ", lpad(@sprintf("%.1f", exp(p1)), 8))
println("Optimum α :  ", lpad(@sprintf("%.1f", α), 8))
println("Optimum γd : ", lpad(@sprintf("%.1f", γd), 8))
println("R2 :         ", lpad(@sprintf("%.1f", 100R2_mech), 8))
text2 = text("γ̂  = " * @sprintf("%.1f", γd) * "\n" *
             "R2 = " * @sprintf("%.1f", 100R2_mech) * " %",
             8, :left)

model = build_constitutive_model(sol_mech.u...)
pl2 = plot()
for i in 1:3:9
  plot_experiment(model, mechanical_data[i], temp_label, pl2)
end
annotate!(pl2, (0.05, 0.72), text2, relative=true)
display(pl2);

pl_ = plot()
cons_model = build_constitutive_model(1.37e4, 5.64e4, log(0.82), 1280.0, 100.0, 0.77, 15.0)
Δt = 0.1
t_values = 0:Δt:10
λ_values = map(1 + 2*triangular(6), t_values)
for T in [0.0, 20.0, 40.0]
  σ_values = simulate_experiment(cons_model, T+K0, Δt, λ_values) / 1e3
  plot!(pl_, λ_values, σ_values, label="T = $T ºC", xlabel="Stretch [-]", ylabel="Stress [kPa]", lw=2)
end
display(pl_);
