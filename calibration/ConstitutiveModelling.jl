using Gridap.TensorValues
using HyperFEM.PhysicalModels, HyperFEM.TensorAlgebra
using HyperFEM.ComputationalModels.EvolutionFunctions

const αr::Float64 = 1.8e-4    # Thermal expansion, /ºK (extracted from 3M VHB technical data sheet)
const K0::Float64 = 273.15    # Celsius to Kelvin conversion
const θr::Float64 = 20.0 + K0 # Reference temperature, ºK
const ϵ0::Float64 = 8.85e-12  # Air permittivity
const t0::Float64 = 0.005     # Specimen thickness, m (5mm)

function J_temp(m::ThermalVolumetric, θ::Float64)
  γ = m.law.γ
  J = 1 + 3*αr*θr/(γ+1)*((θ/θr)^(γ+1)-1)
end

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
  TensorValue(λ, 0, 0, 0, λ2, 0, 0, 0, J/(λ*λ2))
end

function E_t0(V::Float64)
  VectorValue(0.0, V/t0, 0.0)
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
  # Jθ = 1.0 + 3α * (θ - θr)
  Jθ = J_temp(model.thermo, θ)
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

function evaluate_stress(model::ThermoElectroMechano{<:Any,<:Electro,<:ViscoElastic}, Δt, θ, V, λ_values)
  update_time_step!(model, Δt)
  P_func = model()[2]
  n  = length(model.mechano.branches)
  A  = ntuple(_ -> VectorValue(I3..., 0.0), Val(n))
  Jθ = J_temp(model.thermo, θ)
  λ1 = λ_values[1]
  λ2 = sqrt(Jθ)
  Fn = F_vol(λ1, λ2, Jθ)
  E = E_t0(0.0)
  
  function evaluate_P(λ, λ2, E)
    Fi = F_vol(λ, λ2, Jθ)
    Pi = P_func(Fi, E, θ, Fn, A...)
    pi = Pi[2,2] * Fi[2,2]
    Pi = Pi - pi*inv(Fi)
    return Pi, Fi
  end

  function evaluate_P_impl(λ, E)
    P, F = evaluate_P(λ, λ2, E)
    tol = 1e-8
    iter = 0
    maxiter = 20
    while abs(P[3,3]) > tol && iter < maxiter
      δ = 1e-8  # Numerical derivative (secant)
      P_plus, _ = evaluate_P(λ, λ2 + δ, E)
      dP33_dλ2 = (P_plus[3,3] - P[3,3]) / δ
      λ2 -= P[3,3] / dP33_dλ2 # Update λ2
      P, F = evaluate_P(λ, λ2, E)   # Recompute stresses
      iter += 1
    end
    A = new_state(model.mechano, F, Fn, A...)
    Fn = F
    return P, F
  end

  for Vi in range(0.0, V, length=100) # Incrementally apply initial voltage
    E = E_t0(Vi)
    evaluate_P_impl(λ1, E)
  end
  
  map(λ_values) do λ
    P, _ = evaluate_P_impl(λ, E)
    return P[1]
  end
end

evaluate_stress(model::Elasto, θ, λ_values) = evaluate_stress(model, λ_values)

evaluate_stress(model::ViscoElastic, Δt, θ, λ_values) = evaluate_stress(model, Δt, λ_values)

evaluate_stress(model::ThermoMechano{<:Any,<:Elasto}, Δt, θ, λ_values) =  evaluate_stress(model, θ, λ_values)

function evaluate_cv(model::ThermoMechano, θ_values)
  J(θ) = J_temp(mode, θ)
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

function evaluate_epsilon(model::ThermoElectro, θ)
  ∂∂Ψ∂EE = model()[6]
  F1 = F_iso(1.0)
  E0 = E_t0(0.0)
  map(θi -> -1/ϵ0*∂∂Ψ∂EE(F1, E0, θi)[1], θ)
end
