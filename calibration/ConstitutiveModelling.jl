using Gridap.TensorValues
using HyperFEM.PhysicalModels, HyperFEM.TensorAlgebra
using HyperFEM.ComputationalModels.EvolutionFunctions

const αr::Float64 = 1.8e-4    # Thermal expansion, /ºK (extracted from 3M VHB technical data sheet)
const K0::Float64 = 273.15    # Celsius to Kelvin conversion
const θr::Float64 = 20.0 + K0 # Reference temperature, ºK

function F_iso(λ::Float64)
  F_vol(λ, 1.0)
end

function F_vol(J::Float64)
  λ = J^(-1/3)
  TensorValue(λ, 0, 0, 0, λ, 0, 0, 0, λ)
end

function F_vol(λ::Float64, J::Float64)
  # L = sqrt(J/λ)
  # TensorValue(λ, 0, 0, 0, L, 0, 0, 0, L)
  TensorValue(λ, 0, 0, 0, λ^(-1/2), 0, 0, 0, λ^(-1/2)) .* J^(1/3)
end

function F_vol(λ::Float64, λ2::Float64, J::Float64)
  J12 = sqrt(J)
  TensorValue(λ, 0, 0, 0, J12*λ2, 0, 0, 0, J12/(λ*λ2))
end

function new_state(model::ViscoElastic, F, Fn, A...)
  map(model.branches, A) do b, Ai
    _, Se, ∂Se∂Ce = SecondPiola(b.elasto)
    HyperFEM.PhysicalModels.ReturnMapping(b, Se, ∂Se∂Ce, F, Fn, Ai)[2]
  end
end

function evaluate_stress(model::Elasto, λ_values)
  P_func = model()[2]
  map(λ_values) do λ
    F = F_iso(λ)
    P = P_func(F)
    p = P[2,2] * F[2,2]  # Volumetric pressure term
    return P[1] - p / F[1]
  end
end

function evaluate_stress(model::ViscoElastic, Δt, λ_values)
  update_time_step!(model, Δt)
  P_func = model()[2]
  n  = length(model.branches)
  A  = ntuple(_ -> VectorValue(I3..., 0.0), Val(n))
  Fn = F_iso(1.0)
  map(λ_values) do λ
    F = F_iso(λ)
    P = try P_func(F, Fn, A...) catch; zeros(3,3) end
    A = try new_state(model, F, Fn, A...) catch; A end
    Fn = F
    p = P[2,2] * F[2,2]  # Volumetric pressure term
    return P[1] - p / F[1]
  end
end

function evaluate_stress(model::ThermoMechano{<:Any,<:Elasto}, θ, λ_values)
  P_func = model()[2]
  α  = model.thermo.thermo.α
  θr = model.thermo.thermo.θr
  Jθ = 1.0 + α * (θ - θr)
  map(λ_values) do λ
    F = F_vol(λ, Jθ)
    P = P_func(F,θ)
    p = P[2,2] * F[2,2]  # Volumetric pressure term
    return P[1] - p / F[1]
  end
end

function evaluate_stress(model::ThermoMechano{<:Any,<:ViscoElastic}, Δt, θ, λ_values)
  update_time_step!(model, Δt)
  P_func = model()[2]
  n  = length(model.mechano.branches)
  A  = ntuple(_ -> VectorValue(I3..., 0.0), Val(n))
  α  = model.thermo.thermo.α
  θr = model.thermo.thermo.θr
  Jθ = 1.0 + α * (θ - θr)
  Fn = F_vol(1.0, Jθ)
  map(λ_values) do λ
    F = F_vol(λ, Jθ)
    P = P_func(F, θ, Fn, A...)
    p = P[2,2] * F[2,2]
    A = new_state(model.mechano, F, Fn, A...)
    Fn = F
    return P[1] - p / F[1]
  end
end

function evaluate_stress(model::ThermoElectroMechano{<:Any,<:Electro,<:ViscoElastic}, Δt, θ, E2, λ_values)
  update_time_step!(model, Δt)
  P_func = model()[2]
  n  = length(model.mechano.branches)
  A  = ntuple(_ -> VectorValue(I3..., 0.0), Val(n))
  α  = model.thermo.α
  θr = model.thermo.θr
  Jθ = 1.0 + α * (θ - θr)
  E  = VectorValue(0, E2, 0)
  Fn = F_vol(1.0, Jθ)
  λ2 = 1.0
  map(λ_values) do λ
    function P_total(λ1, λ2)
      F = F_vol(λ1, λ2, Jθ)
      P = P_func(F, E, θ, Fn, A...)
      p = P[2,2] * F[2,2]
      P = P - p*inv(F)
      return (P[1,1], P[2,2], P[3,3])
    end
    P = P_total(λ, λ2)
    while abs(P[3]) > 1e-6

    end
    A = new_state(model.mechano, F, Fn, A...)
    Fn = F
    return P[1] - p / F[1]
  end
end

evaluate_stress(model::Elasto, θ, λ_values) = evaluate_stress(model, λ_values)

evaluate_stress(model::ViscoElastic, Δt, θ, λ_values) = evaluate_stress(model, Δt, λ_values)

evaluate_stress(model::ThermoMechano{<:Any,<:Elasto}, Δt, θ, λ_values) =  evaluate_stress(model, θ, λ_values)

function evaluate_cv(model::ThermoMechano, θ_values)
  γ = model.law.γ
  J(θ) = 1 + 3*αr*θr/(γ+1)*((θ/θr)^(γ+1)-1)
  ∂∂Ψ = model()[5]
  if model.mechano isa Elasto
    return map(θ -> -θ*∂∂Ψ(F_vol(J(θ)), θ), θ_values)
  else
    update_time_step!(model, 1.0)
    n = length(model.mechano.branches)
    A = ntuple(_ -> VectorValue(I3..., 0.0), Val(n))
    return map(θ -> -θ*∂∂Ψ(F_vol(J(θ)), θ, F_vol(J(θ)), A...), θ_values)
  end
end

function evaluate_cv(model::ThermoMechano, θ, λ, v)
  steps = 20
  n  = length(model.mechano.branches)
  F0 = F_iso(1.0)
  A  = ntuple(_ -> VectorValue(I3..., 0.0), Val(n))
  Δt = abs(λ - 1) / v / steps
  update_time_step!(model, Δt)
  ∂∂Ψ∂θθ = model()[5]
  cv(F,θ,X...) = -θ*∂∂Ψ∂θθ(F,θ,X...)
  if λ ≈ 1.0
    update_time_step!(model, 1.0)
    return cv(F0, θ, F0, A...)
  else
    for λi ∈ range(1, λ, steps+1)  # First step is static -> n+1
      Fi = F_iso(λi)
      if λi ≈ λ  # we compute and return the specific heat at the last time step
        return cv(Fi, θ, F0, A...)
      end
      A = new_state(model.mechano, Fi, F0, A...)
      F0 = Fi
    end
  end
end
