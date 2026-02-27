using Gridap.TensorValues
using HyperFEM.PhysicalModels, HyperFEM.TensorAlgebra
using HyperFEM.ComputationalModels.EvolutionFunctions

Base.broadcastable(m::T) where {T<:PhysicalModel} = Ref(m) # Allow to use the @. syntax for passing a single constitutive model into a vectorized functions

const ρr::Float64 = 720.0     # Density, kg/m³ (extracted from 3M VHB technical data sheet)
const αr::Float64 = 1.8e-4    # Thermal expansion, /ºK (extracted from 3M VHB technical data sheet)
const K0::Float64 = 273.15    # Celsius to Kelvin conversion
const θr::Float64 = 20.0 + K0 # Reference temperature, ºK

function F_iso(λ::Float64)
  F_vol(λ, 1.0)
end

function F_vol(λ::Float64, J::Float64)
  g = sqrt(J/λ)
  TensorValue(λ, 0, 0, 0, g, 0, 0, 0, g)
end

function new_state(model::ViscoElastic, F, Fn, A...)
  n = length(model.branches)
  map(1:n) do i
    b = model.branches[i]
    _, Se, ∂Se∂Ce = SecondPiola(b.elasto)
    HyperFEM.PhysicalModels.ReturnMapping(b, Se, ∂Se∂Ce, F, Fn, A[i])[2]
  end
end

function evaluate_stress(model::Elasto, λ_values)
  P = model()[2]
  map(λ_values) do λ
    F = F_iso(λ)
    σ = P(F)[1]
    return σ
  end
end

function evaluate_stress(model::ViscoElastic, Δt, λ_values)
  update_time_step!(model, Δt)
  n  = length(model.branches)
  P  = model()[2]
  A  = fill(VectorValue(I3..., 1), n)
  Fn = F_iso(1.0)
  map(λ_values) do λ
    F = F_iso(λ)
    σ = P(F, Fn, A...)[1]
    A = new_state(model, F, Fn, A...)
    Fn = F
    return σ
  end
end

function evaluate_stress(model::ThermoMechano, Δt, θ, λ_values)
  update_time_step!(model, Δt)
  n  = length(model.mechano.branches)
  P  = model()[2]
  A  = fill(VectorValue(I3..., 0), n)
  α  = model.thermo.α
  θr = model.thermo.θr
  Jθ = 1.0 + α * (θ - θr)
  Fn = F_vol(1.0, Jθ)
  σ0 = P(Fn, θ, Fn, A...)[1]
  map(λ_values) do λ
    F = F_vol(λ, Jθ)
    σ = P(F, θ, Fn, A...)[1] - σ0
    A = new_state(model.mechano, F, Fn, A...)
    Fn = F
    return σ
  end
end

evaluate_stress(model::Elasto, θ, λ_values) = evaluate_stress(model, λ_values)

evaluate_stress(model::ViscoElastic, Δt, θ, λ_values) = evaluate_stress(model, Δt, λ_values)

function evaluate_cv(model::ThermoMechano, θ_values)
  ∂∂Ψ = model()[5]
  F   = F_iso(1.0)
  if model.mechano isa Elasto
    return map(θ -> -θ*∂∂Ψ(F, θ), θ_values)
  else
    update_time_step!(model, 1.0)
    n   = length(model.mechano.branches)
    A   = fill(VectorValue(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0), n)
    return map(θ -> -θ*∂∂Ψ(F, θ, F, A...), θ_values)
  end
end

function evaluate_cv(model::ThermoMechano, θ, λ, v)
  n  = length(model.mechano.branches)
  F0 = F_iso(1.0)
  A  = fill(VectorValue(I3..., 0.0), n)
  Δt = abs(λ - 1) / v / 10
  update_time_step!(model, Δt)
  ∂∂Ψ∂θθ = model()[5]
  cv(F,θ,X...) = -θ*∂∂Ψ∂θθ(F,θ,X...)
  if λ ≈ 1.0
    update_time_step!(model, 1.0)
    return cv(F0, θ, F0, A...)
  else
    for λi ∈ range(1, λ, 11)  # perform a substepping
      Fi = F_iso(λi)
      if λi ≈ λ  # we compute and return the specific heat at the last time step
        return cv(Fi, θ, F0, A...)
      end
      A = new_state(model.mechano, Fi, F0, A...)
      F0 = Fi
    end
  end
end

simulate_experiment(model::ViscoElastic, θ, Δt, λ_values) = simulate_experiment(model, Δt, λ_values)

function simulate_experiment(model::ViscoElastic, Δt, λ_values)
  update_time_step!(model, Δt)
  n  = length(model.branches)
  P  = model()[2]
  A  = fill(VectorValue(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0), n)
  Fn = F_iso(1.0)
  map(λ_values) do λ
    F = F_iso(λ)
    σ = P(F, Fn, A...)[1]
    A = new_state(model, F, Fn, A...)
    Fn = F
    return σ
  end
end

simulate_experiment(model::Elasto, θ, λ_values) = simulate_experiment(model, λ_values)

function simulate_experiment(model::Elasto, λ_values)
  P = model()[2]
  map(λ_values) do λ
    F = F_iso(λ)
    σ = P(F)[1]
    return σ
  end
end

function simulate_experiment(model::ThermoMechano, θ, Δt, λ_values)
  update_time_step!(model, Δt)
  n  = length(model.mechano.branches)
  P  = model()[2]
  A  = fill(VectorValue(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0), n)
  α  = model.thermo.α
  θr = model.thermo.θr
  Jθ = 1.0 + α * (θ - θr)
  Fn = F_vol(1.0, Jθ)
  σ0 = P(Fn, θ, Fn, A...)[1]
  map(λ_values) do λ
    F = F_vol(λ, Jθ)
    σ = P(F, θ, Fn, A...)[1] - σ0
    A = new_state(model.mechano, F, Fn, A...)
    Fn = F
    return σ
  end
end

function simulate_experiment(model::ThermoMechano, θ_values)
  ∂∂Ψ = model()[5]
  F   = F_iso(1.0)
  if model.mechano isa Elasto
    return map(θ -> -θ*∂∂Ψ(F, θ), θ_values)
  else
    update_time_step!(model, 1.0)
    n   = length(model.mechano.branches)
    A   = fill(VectorValue(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0), n)
    return map(θ -> -θ*∂∂Ψ(F, θ, F, A...), θ_values)
  end
end

function cv_single_step_stretch(model::ThermoMechano, λ, θ, v)
  n  = length(model.mechano.branches)
  F0 = F_iso(1.0)
  F1 = F_iso(λ)
  A  = fill(VectorValue(I3..., 0.0), n)
  Δt = abs(λ - 1) / v / 10
  update_time_step!(model, Δt)
  ∂∂Ψ∂θθ = model()[5]
  cv(F,θ,X...) = -θ*∂∂Ψ∂θθ(F,θ,X...)
  if λ ≈ 1.0
    update_time_step!(model, 1.0)
    return cv(F1, θ, F0, A...)
  else
    for λi ∈ range(1+(λ-1)/10, λ, 10)  # perform a substepping
      Fi = F_iso(λi)
      if λi ≈ λ  # we compute and return the specific heat at the last time step
        return cv(Fi, θ, F0, A...)
      end
      A = new_state(model.mechano, Fi, F0, A...)
      F0 = Fi
    end
  end
end
