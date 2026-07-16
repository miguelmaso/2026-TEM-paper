using JLD2

include("Membrane.jl")
include("../AcousticTensor/AcousticTensor.jl")

step = 10

@load joinpath(@__DIR__, "results/Membrane_state_$(step).jld2") time Fq⁺ Fq⁻ Eq θq Aq


model = build_model(θr=293.15)
update_time_step!(model, 0.02)

H_FF = model()[5]
H_EF = model()[8]
H_θF = model()[9]

Hq_FF = H_FF(Fq⁺, Eq, θq, Fq⁻, Aq...)
Hq_EF = H_EF(Fq⁺, Eq, θq, Fq⁻, Aq...)
Hq_θF = H_θF(Fq⁺, Eq, θq, Fq⁻, Aq...)

surface_plot(acoustic_tensor_positiveness(Hq_FF))
polar_plot(acoustic_tensor_positiveness(Hq_FF))

polar_plot(H_FF_bulk(Hq_FF))
polar_plot(H_EF_elec(Hq_EF))
polar_plot(H_θF_therm(Hq_θF))


surface_plot(H_EF_elec(Hq_EF))
