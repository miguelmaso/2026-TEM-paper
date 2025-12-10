#------------------------------------------
#------------------------------------------
# 
# Relaxation benchmark
# Mokarram et al., 2012, Experimental study and numerical modelling of VHB 4910 polymer
# Liu et al., 2025, Large strain constitutive modelling of soft compressible and incompressible solids: Generalised isotropic and isotropic viscoelasticity
# 
#------------------------------------------
#------------------------------------------

using Gridap
using HyperFEM
using Plots
using HyperFEM.ComputationalModels.EvolutionFunctions

#------------------------------------------
# Visco-elastic model
#------------------------------------------
μ  = 1.37e4  # Pa
μ1 = 3.15e4  # Pa
τ1 = 10.7    # s
μ2 = 5.64e4  # Pa
τ2 = 0.82    # s
μ3 = 1.98e4  # Pa
τ3 = 500.0   # s
hyper_elastic_model = NeoHookean3D(λ=100μ, μ=μ)
branch1 = ViscousIncompressible(IncompressibleNeoHookean3D(λ=0., μ=μ1), τ=τ1)
branch2 = ViscousIncompressible(IncompressibleNeoHookean3D(λ=0., μ=μ2), τ=τ2)
branch3 = ViscousIncompressible(IncompressibleNeoHookean3D(λ=0., μ=μ3), τ=τ3)
cons_model = GeneralizedMaxwell(hyper_elastic_model, branch1, branch2, branch3)

function F_iso(λ::Float64)
  f = 1 + λ
  TensorValue(f, 0, 0, 0, f^-.5, 0, 0, 0, f^-.5)
end

function new_state(F, Fn, A...)
    map(1:3) do i
        b = cons_model.branches[i]
        _, Se, ∂Se∂Ce = SecondPiola(b.elasto)
        HyperFEM.PhysicalModels.ReturnMapping(b, Se, ∂Se∂Ce, F, Fn, A[i])[2]
    end
end

function experimental_test(λ_values, Δt)
  update_time_step!(cons_model, Δt)
  P  = cons_model()[2]
  A  = fill(VectorValue(1, 0, 0, 0, 1, 0, 0, 0, 1, 0), 3)
  Fn = F_iso(0.0)
  map(λ_values) do λ
    F = F_iso(λ)
    σ = P(F, Fn, A...)[1]
    A  = new_state(F, Fn, A...)
    Fn = F
    return σ
  end
end

# ----------------------------------
# Single-step relaxation test
# ----------------------------------
Δt = 1.0
times = 0:Δt:600

λ02 = map(t -> 0.2*ramp(0.2)(t), times)
σ02 = experimental_test(λ02, Δt) / 1e3

λ04 = map(t -> 0.4*ramp(0.4)(t), times)
σ04 = experimental_test(λ04, Δt) / 1e3

p1 = plot([times, times.+100], [λ02 λ04], labels=["Exper 20%" "Exper 40%"], lw=2, xlabel="Time [s]", ylabel="Strain [-]")
p2 = plot([times, times.+100], [σ02 σ04], labels=["Exper 20%" "Exper 40%"], lw=2, xlabel="Time [s]", ylabel="Stress [kPa]")
p = plot(p1, p2)
display(p)
savefig(p, joinpath(@__DIR__, "single-step.png"))

# ----------------------------------
# Loading-unloading test
# ----------------------------------
Δt = 1.0

t001 = 0:Δt:400
λ001 = map(t -> 2*triangular(200.)(t), t001)
σ001 = experimental_test(λ001, Δt) / 1e3

t005 = 0:Δt:80
λ005 = map(t -> 2*triangular(40.)(t), t005)
σ005 = experimental_test(λ005, Δt) / 1e3

p1 = plot([t001, t005], [λ001, λ005], labels=["ε'=0.01 s⁻¹" "ε'=0.05 s⁻¹"], lw=2, xlabel="Time [s]", ylabel="Strain [-]")
p2 = plot([λ001, λ005], [σ001, σ005], labels=["ε'=0.01 s⁻¹" "ε'=0.05 s⁻¹"], lw=2, xlabel="Strain [-]", ylabel="Stress [kPa]", ylims=[0,Inf])
p = plot(p1, p2)
display(p)
savefig(p, joinpath(@__DIR__, "loading-unloading.png"))
