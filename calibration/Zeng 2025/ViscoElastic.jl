#------------------------------------------
#------------------------------------------
#
# Relaxation benchmark
# Mokarram et al., 2012, Experimental study and numerical modelling of VHB 4910 polymer
# Liu et al., 2025, Large strain constitutive modelling of soft compressible and incompressible solids: Generalised isotropic and isotropic viscoelasticity
#
#------------------------------------------
#------------------------------------------

using Plots, Printf
using HyperFEM, HyperFEM.ComputationalModels.EvolutionFunctions

include("../ConstitutiveModelling.jl")
include("../ExperimentsData.jl")

#------------------------------------------
# Visco-elastic model
# (Table 1: Identified viscous parameters using experimental data at deformation level of 50%)
#------------------------------------------
μ  = 1.37e4  # Pa
N  = 7.86e5  # -
μ1 = 5.64e4  # Pa
τ1 = 0.82    # s
μ2 = 3.15e4  # Pa
τ2 = 10.7    # s
μ3 = 1.98e4  # Pa
τ3 = 500.0   # s
hyper_elastic_model = EightChain(μ=μ, N=N)
branch1 = ViscousIncompressible(IncompressibleNeoHookean3D(λ=0., μ=μ1), τ=τ1)
branch2 = ViscousIncompressible(IncompressibleNeoHookean3D(λ=0., μ=μ2), τ=τ2)
branch3 = ViscousIncompressible(IncompressibleNeoHookean3D(λ=0., μ=μ3), τ=τ3)
cons_model = GeneralizedMaxwell(hyper_elastic_model, branch1, branch2, branch3)

# ----------------------------------
# Single-step relaxation test
# ----------------------------------
Δt = 1.0
times = 0:Δt:600

λ02 = map(1 + 0.2*ramp(0.2), times)
σ02 = evaluate_stress(cons_model, Δt, λ02) / 1e3

λ04 = map(1 + 0.4*ramp(0.4), times)
σ04 = evaluate_stress(cons_model, Δt, λ04) / 1e3

p1 = plot([times, times.+100], [λ02 λ04], labels=["Exper 20%" "Exper 40%"], lw=2, xlabel="Time [s]", ylabel="Strain [-]")
p2 = plot([times, times.+100], [σ02 σ04], labels=["Exper 20%" "Exper 40%"], lw=2, xlabel="Time [s]", ylabel="Stress [kPa]")
p = plot(p1, p2)
display(p)
savefig(p, joinpath(@__DIR__, "single-step.png"))

# ----------------------------------
# Loading-unloading test
# ----------------------------------
Δt = 1.0

v = 0.01
t001 = 0:Δt:2*1/v
λ001 = map(1 + 1*triangular(1/v), t001)
σ001 = evaluate_stress(cons_model, Δt, λ001) / 1e3

v = 0.05
t005 = 0:Δt:2*1/v
λ005 = map(1 + 1*triangular(1/v), t005)
σ005 = evaluate_stress(cons_model, Δt, λ005) / 1e3

p1 = plot([t001, t005], [λ001, λ005], labels=["v=0.01 s⁻¹" "v=0.05 s⁻¹"], lw=2, xlabel="Time [s]", ylabel="Strain [-]")
p2 = plot([λ001, λ005], [σ001, σ005], labels=["v=0.01 s⁻¹" "v=0.05 s⁻¹"], lw=2, xlabel="Strain [-]", ylabel="Stress [kPa]", ylims=[0,Inf])
p = plot(p1, p2)
display(p)
savefig(p, joinpath(@__DIR__, "loading-unloading.png"))
