#------------------------------------------
#------------------------------------------
# 
# Calibration of VHB 4905 polymer
# Liao, Mokarram et al., 2022, On thermo-viscoelastic experimental characterization and numerical modelling of VHB polymer
# 
#------------------------------------------
#------------------------------------------

using CSV, DataFrames
using Gridap.TensorValues
using Plots
using Statistics
using Optimization, OptimizationOptimJL, ForwardDiff, OptimizationMetaheuristics
using HyperFEM.PhysicalModels, HyperFEM.TensorAlgebra
using HyperFEM.ComputationalModels.EvolutionFunctions

# The header of the csv must contain the following fields:
#   - id
#   - temp
#   - stretch
#   - dt
#   - stress

struct ExperimentData
  id::Int
  θ::Float64
  Δt::Float64
  λ::Vector{Float64}
  σ::Vector{Float64}
  σ_max::Float64
  weight::Float64
end

function ExperimentData(df, weight = 1.0)
  id = df.id[1]
  θ  = df.temp[1]
  Δt = mean(df.dt)
  λ  = df.stretch
  σ  = df.stress * 1e6 # Input values are in MPa
  σ_max = maximum(abs.(σ))
  ExperimentData(id, θ, Δt, λ, σ, σ_max, weight)
end

function load_data(filepath::String)
  df = CSV.read(filepath, DataFrame; decimal=',')
  grouped = groupby(df, :id)
  experiments = Vector{ExperimentData}()
  for sub_df ∈ grouped
    push!(experiments, ExperimentData(sub_df))
  end
  experiments
end

function F_iso(λ::Float64)
  TensorValue(λ, 0, 0, 0, λ^-.5, 0, 0, 0, λ^-.5)
end

function new_state(model, F, Fn, A...)
  n = length(model.branches)
  map(1:n) do i
    b = model.branches[i]
    _, Se, ∂Se∂Ce = SecondPiola(b.elasto)
    HyperFEM.PhysicalModels.ReturnMapping(b, Se, ∂Se∂Ce, F, Fn, A[i])[2]
  end
end

function simulate_experiment(model, θ, Δt, λ_values)
  update_time_step!(model, Δt)
  n = length(model.mechano.branches)
  P  = model()[2]
  A  = fill(VectorValue(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0), n)
  Fn = F_iso(1.0)
  map(λ_values) do λ
    F = F_iso(λ)
    σ = P(F, θ, Fn, A...)[1]
    A = new_state(model.mechano, F, Fn, A...)
    Fn = F
    return σ
  end
end

function loss(model::PhysicalModel, data::Vector{ExperimentData})
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

# Load the experiments data
experiments_data = load_data(joinpath(@__DIR__, "Liao_Mokarram.csv"))

long_term = NeoHookean3D(λ=1.0e6, μ=1.37e4)
branch_1 = ViscousIncompressible(IncompressibleNeoHookean3D(λ=0., μ=5.64e4), τ=0.82)
visco_elasto = GeneralizedMaxwell(long_term, branch_1)
thermal_model = ThermalModel3rdLaw(cv0=10.0, θr=293.15, α=1.0, κ=1.0, γv=0.5, γd=0.5)
cons_model = ThermoMechModel(thermal_model, visco_elasto)

e1 = experiments_data[1]
σ1 = simulate_experiment(cons_model, 293.15, e1.Δt, e1.λ)
plot(e1.λ, σ1)

# Initial seed
p0 = [1000.0, 1000.0, log(0.2), 1.0, 0.5, 0.5]

# Search limits
lb = [100.0,  100.0, -5.0,  0.0, 0.0, 0.0]  # Minimums
ub = [5.0e4, 10.0e4,  5.0, 10.0, 1.0, 1.0]  # Maximums

# Definition of the objective function
opt_func = OptimizationFunction(global_loss)   # AutoFiniteDiff()
opt_prob = OptimizationProblem(opt_func, p0, experiments_data, lb=lb, ub=ub)

# Optimal parameters. ECA (Evolutionary Centers Algorithm)
sol = solve(opt_prob, ECA(), maxiters=10000, maxtime=60.0)  # ECA, NelderMead, LBFGS
println("p optimo: ", sol.u)
println("Error final: ", sol.minimum)
