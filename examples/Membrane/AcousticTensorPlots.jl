using JLD2

include("Membrane.jl")
include("AcousticTensorDefinitions.jl")

step = 25

@load joinpath(@__DIR__, "results/Membrane_state_$(step).jld2") time Fq⁺ Fq⁻ Eq θq Aq

@show time

model = build_model(θr=293.15)
update_time_step!(model, 0.02)

Ψ, ∂Ψ_F, ∂Ψ_E, ∂Ψ_θ, ∂∂Ψ_FF, ∂∂Ψ_EE, ∂∂Ψ_θθ, ∂∂Ψ_EF, ∂∂Ψ_Fθ, ∂∂Ψ_Eθ = model()

∂∂Ψq_FF = ∂∂Ψ_FF(Fq⁺, Eq, θq, Fq⁻, Aq...)
∂∂Ψq_θθ = ∂∂Ψ_FF(Fq⁺, Eq, θq, Fq⁻, Aq...)
∂∂Ψq_EF = ∂∂Ψ_EF(Fq⁺, Eq, θq, Fq⁻, Aq...)
∂∂Ψq_Fθ = ∂∂Ψ_Fθ(Fq⁺, Eq, θq, Fq⁻, Aq...)
∂∂Ψq_Eθ = ∂∂Ψ_Eθ(Fq⁺, Eq, θq, Fq⁻, Aq...)

∂∂Wq_FF = ∂∂Ψq_FF - ∂∂Ψq_θθ^(-1) * (∂∂Ψq_Fθ ⊗ ∂∂Ψq_Fθ)  # NOTA 1: F/θ permutados    NOTA 2: Sigue siendo respecto E0 en lugar de D0, multiplicar por εr???
∂∂Wq_EF = ∂∂Ψq_EF - ∂∂Ψq_θθ^(-1) * (∂∂Ψq_Eθ ⊗ ∂∂Ψq_Fθ)  # NOTA 1: F/θ permutados    NOTA 2: Sigue siendo respecto E0 en lugar de D0, multiplicar por εr???
∂∂Wq_Fη = -∂∂Ψq_θθ^(-1) * ∂∂Ψq_Fθ                        # NOTA 2: Sigue siendo respecto E0 en lugar de D0, multiplicar por εr???


surface_plot(acoustic_tensor_positiveness(Hq_FF), joinpath(@__DIR__, "fig/acoustic_tensor_$(step)_FF_positiveness.png"))

polar_plot(H_FF_bulk(Hq_FF),    joinpath(@__DIR__, "fig/acoustic_tensor_$(step)_FF_bulk.png"))
polar_plot(H_FF_shear_α(Hq_FF), joinpath(@__DIR__, "fig/acoustic_tensor_$(step)_FF_shear_1.png"))
polar_plot(H_FF_shear_β(Hq_FF), joinpath(@__DIR__, "fig/acoustic_tensor_$(step)_FF_shear_2.png"))
polar_plot(H_EF_elec(Hq_EF),    joinpath(@__DIR__, "fig/acoustic_tensor_$(step)_EF.png"))
polar_plot(H_θF_therm(Hq_θF),   joinpath(@__DIR__, "fig/acoustic_tensor_$(step)_TF.png"))
