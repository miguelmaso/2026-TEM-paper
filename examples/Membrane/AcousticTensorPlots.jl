using JLD2

include("Membrane.jl")
include("../AcousticTensor/AcousticTensor.jl")

step = 0

@load joinpath(@__DIR__, "results/Membrane_state_$(step).jld2") time Fq⁺ Fq⁻ Eq θq Aq

@show time

model = build_model(θr=293.15)
update_time_step!(model, 0.02)

H_FF = model()[5]
H_EF = model()[8]
H_θF = model()[9]

Hq_FF = H_FF(Fq⁺, Eq, θq, Fq⁻, Aq...)
Hq_EF = H_EF(Fq⁺, Eq, θq, Fq⁻, Aq...)
Hq_θF = H_θF(Fq⁺, Eq, θq, Fq⁻, Aq...)

surface_plot(acoustic_tensor_positiveness(Hq_FF))

polar_plot(acoustic_tensor_positiveness(Hq_FF), joinpath(@__DIR__, "fig/acoustic_tensor_$(step)_FF_positiveness.png"))
polar_plot(H_FF_bulk(Hq_FF),    joinpath(@__DIR__, "fig/acoustic_tensor_$(step)_FF_bulk.png"))
polar_plot(H_FF_shear_α(Hq_FF), joinpath(@__DIR__, "fig/acoustic_tensor_$(step)_FF_shear_1.png"))
polar_plot(H_FF_shear_β(Hq_FF), joinpath(@__DIR__, "fig/acoustic_tensor_$(step)_FF_shear_2.png"))
polar_plot(H_EF_elec(Hq_EF),    joinpath(@__DIR__, "fig/acoustic_tensor_$(step)_EF.png"))
polar_plot(H_θF_therm(Hq_θF),   joinpath(@__DIR__, "fig/acoustic_tensor_$(step)_TF.png"))
