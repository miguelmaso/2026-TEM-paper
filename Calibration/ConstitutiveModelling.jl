using Gridap.TensorValues
using HyperFEM.PhysicalModels, HyperFEM.TensorAlgebra
using HyperFEM.ComputationalModels.EvolutionFunctions

function F_iso(λ::Float64)
  TensorValue(λ, 0, 0, 0, λ^-.5, 0, 0, 0, λ^-.5)
end

function new_state(model::ViscoElastic, F, Fn, A...)
  n = length(model.branches)
  map(1:n) do i
    b = model.branches[i]
    _, Se, ∂Se∂Ce = SecondPiola(b.elasto)
    HyperFEM.PhysicalModels.ReturnMapping(b, Se, ∂Se∂Ce, F, Fn, A[i])[2]
  end
end

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

function simulate_experiment(model::ThermoMechano, θ, Δt, λ_values)
  update_time_step!(model, Δt)
  n  = length(model.mechano.branches)
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

function simulate_experiment(model::ThermoMechano, θ_values)
  update_time_step!(model, 1.0)
  n   = length(model.mechano.branches)
  ∂∂Ψ = model()[6]
  A   = fill(VectorValue(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0), n)
  F   = F_iso(1.0)
  map(θ -> -1.0*∂∂Ψ(F, θ, F, A...), θ_values)
end
