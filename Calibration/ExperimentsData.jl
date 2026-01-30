using CSV, DataFrames
using Statistics

abstract type ExperimentData end

struct LoadingTest <: ExperimentData
  id::Int
  θ::Float64
  v::Float64
  Δt::Float64
  λ::Vector{Float64}
  σ::Vector{Float64}
  λ_max::Float64
  σ_max::Float64
  weight::Float64

  function LoadingTest(df, weight=1.0)
    id = df.id[1]
    θ  = df.temp[1]
    Δt = mean(df.dt)
    λ  = df.stretch
    σ  = df.stress * 1e6 # Input values are in MPa
    σ_max = maximum(abs.(σ))
    λ_max = round(maximum(abs.(λ)))
    v     = round(λ[5] / 5 / Δt)
    new(id, θ, v, Δt, λ, σ, λ_max, σ_max, weight)
  end
end

struct HeatingTest <: ExperimentData
  id::Int
  θ::Vector{Float64}
  cv::Vector{Float64}
  cv_max::Float64
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
