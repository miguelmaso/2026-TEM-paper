using CSV, DataFrames
using Statistics

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
    v     = round(λ[5] / 5 / Δt)
    new(id, θ, v, Δt, λ, σ, λ_max, σ_max, weight)
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
end

function read_data(filepath::String, experiment_type::Type)
  df = CSV.read(filepath, DataFrame; decimal=',')
  grouped = groupby(df, :id)
  experiments = Vector{experiment_type}()
  for sub_df ∈ grouped
    push!(experiments, experiment_type(sub_df))
  end
  experiments
end
