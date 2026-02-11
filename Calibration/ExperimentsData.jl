using CSV, DataFrames
using Statistics
using Parameters

abstract type ExperimentData end

mutable struct LoadingTest <: ExperimentData
  const id::Int
  const θ::Float64
  const v::Float64
  const Δt::Float64
  const λ::Vector{Float64}
  const σ::Vector{Float64}
  const λ_max::Float64
  const σ_max::Float64
  weight::Float64

  function LoadingTest(df, weight=1.0)
    id = df.id[1]
    θ  = df.temp[1]
    Δt = mean(df.dt)
    λ  = df.stretch
    σ  = df.stress * 1e6 # Input values are in MPa, converted into Pa
    σ_max = maximum(abs.(σ))
    λ_max = round(maximum(abs.(λ)))
    i_max = argmax(λ)-1
    v     = round((λ[i_max]-λ[1]) / (i_max-1) / Δt; digits=2)
    new(id, θ, v, Δt, λ, σ, λ_max, σ_max, weight)
  end

  function LoadingTest(other::LoadingTest, new_σ)
    @unpack id, θ, v, Δt, λ, σ, λ_max, σ_max, weight = other
    new(id, θ, v, Δt, λ, new_σ, λ_max, σ_max, weight)
  end
end

mutable struct HeatingTest <: ExperimentData
  const id::Int
  const θ::Vector{Float64}
  const cv::Vector{Float64}
  const cv_max::Float64
  weight::Float64

  function HeatingTest(df, weight=1.0)
    id = df.id[1]
    θ  = df.temp
    cv = df.cv
    cv_max = maximum(abs.(cv))
    new(id, θ, cv, cv_max, weight)
  end

  function HeatingTest(other::HeatingTest, new_cv)
    @unpack id, θ, cv, cv_max, weight = other
    new(id, θ, new_cv, cv_max, weight)
  end
end

npoints(test::LoadingTest) = length(test.λ)

npoints(test::HeatingTest) = length(test.θ)

npoints(tests::Vector{<:ExperimentData}) = sum(npoints, tests)

function read_data(filepath::String, experiment_type::Type)
  df = CSV.read(filepath, DataFrame; decimal=',')
  grouped = groupby(df, :id)
  experiments = Vector{experiment_type}()
  for sub_df ∈ grouped
    push!(experiments, experiment_type(sub_df))
  end
  experiments
end

function Base.print(ds::Vector{HeatingTest})
  println("_id_|___cv___|__w_")
  foreach(r -> println(
      @sprintf("%3d | ", r.id) *
      @sprintf("%.1f | ", r.cv_max) *
      @sprintf("%.1f", r.weight)
  ), ds)
end

function Base.print(ds::Vector{LoadingTest})
  println("_id_|___T___|__λ__|___v__|__w_")
  foreach(r -> println(
      @sprintf("%3d | ", r.id) *
      @sprintf("%3.0fºC | ", r.θ-K0) *
      @sprintf("%.1f | ", r.λ_max) *
      @sprintf("%.2f | ", r.v) *
      @sprintf("%.1f", r.weight)
    ), ds)
end

function Base.println(ds::Vector{<:ExperimentData})
  print(ds)
  print("\n")
end

getfirst(pred,itr) = first(Iterators.filter(pred,itr))
