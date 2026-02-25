using CSV, DataFrames
using Statistics
using Parameters

abstract type ExperimentData end
abstract type MechanicalTest <: ExperimentData end
abstract type ThermalTest <: ExperimentData end

mutable struct LoadingTest <: MechanicalTest
  const id::Int
  const θ::Float64
  const v::Float64
  const Δt::Float64
  const λ::Vector{Float64}
  const σ::Vector{Float64}
  const λ_max::Float64
  const σ_max::Float64
  weight::Float64
end

function LoadingTest(df, weight=1.0)
  id = df.id[1]
  θ  = df.temp[1]
  v  = df.vel[1]
  λ  = df.stretch
  σ  = df.stress
  σ_max = maximum(abs.(σ))
  λ_max = round(maximum(abs.(λ)))  # It is expected to be 2.0, 3.0 or 4.0
  i_max = argmax(λ)-1
  Δt    = (λ[i_max]-λ[1]) / (i_max-1) / v
  LoadingTest(id, θ, v, Δt, λ, σ, λ_max, σ_max, weight)
end

mutable struct CreepTest <: MechanicalTest
  const id::Int
  const θ::Float64
  const Δt::Float64
  const λ_max::Float64
  const t::Vector{Float64}
  const σ::Vector{Float64}
  weight::Float64
end

function CreepTest(df, weight=1.0)
  id = df.id[1]
  θ  = df.temp[1]
  λ_max = df.stretch[1]
  t  = df.time
  σ  = df.stress
  Δt = t[end] / length(t)
  CreepTest(id, θ, Δt, λ_max, t, σ, weight)
end

mutable struct QuasiStaticTest <: MechanicalTest
  const id ::Int
  const θ::Float64
  const λ::Vector{Float64}
  const σ::Vector{Float64}
  weight::Float64
end

function QuasiStaticTest(df, weight=1.0)
  id = df.id[1]
  θ  = df.temp[1]
  λ  = df.stretch
  σ  = df.stress
  QuasiStaticTest(id, θ, λ, σ, weight)
end

mutable struct CalorimetryTest <: ThermalTest
  const id::Int
  const θ::Vector{Float64}
  const cv::Vector{Float64}
  const cv_max::Float64
  weight::Float64
end

function CalorimetryTest(df, weight=1.0)
  id = df.id[1]
  θ  = df.temp
  cv = df.cv
  cv_max = maximum(abs.(cv))
  CalorimetryTest(id, θ, cv, cv_max, weight)
end

npoints(test::LoadingTest) = length(test.λ)

npoints(test::CreepTest) = length(test.t)

npoints(test::QuasiStaticTest) = length(test.λ)

npoints(test::CalorimetryTest) = length(test.θ)

npoints(tests::Vector{<:ExperimentData}) = sum(npoints, tests)

function load_data(filepath::String, experiment_type::Type)
  df = CSV.read(filepath, DataFrame; decimal=',')
  grouped = groupby(df, :id)
  experiments = Vector{experiment_type}()
  for sub_df ∈ grouped
    push!(experiments, experiment_type(sub_df))
  end
  experiments
end

function Base.print(data::Vector{CalorimetryTest})
  println("Set of $(length(data)) $(CalorimetryTest)")
  println("__id_|____cv___|__w_")
  foreach(r -> @printf(
      "%4d | %7.2g | %.1f\n",
      r.id, r.cv_max, r.weight
    ), data)
end

function Base.print(data::Vector{LoadingTest})
  println("Set of $(length(data)) $(LoadingTest)")
  println("__id_|___T___|__λ__|___v__|__w_")
  foreach(r -> @printf(
      "%4d | %3.0fºC | %.1f | %.2f | %.1f\n",
      r.id, r.θ-K0, r.λ_max, r.v, r.weight
    ), data)
end

function Base.print(data::Vector{CreepTest})
  println("Set of $(length(data)) $(CreepTest)")
  println("__id_|___T___|__λ__|__w_")
  foreach(r -> @printf(
      "%4d | %3.0fºC | %.1f | %.1f\n",
      r.id, r.θ-K0, r.λ_max, r.weight
    ), data)
end

function Base.print(data::Vector{QuasiStaticTest})
  println("Set of $(length(data)) $(QuasiStaticTest)")
  println("__id_|___T___|__w_")
  foreach(r -> @printf(
      "%4d | %3.0fºC | %.1f\n",
      r.id, r.θ-K0, r.weight
    ), data)
end

function Base.println(data::Vector{<:ExperimentData})
  print(data)
  print("\n")
end

getfirst(pred,itr) = first(Iterators.filter(pred,itr))
