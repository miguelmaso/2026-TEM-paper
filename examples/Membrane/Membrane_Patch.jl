using HyperFEM
using Gridap
using NonlinearSolve
using Printf
using Plots
import Plots:mm

pname = stem(@__FILE__)
folder = abspath(dirname(@__FILE__), "results")
outpath = joinpath(folder, pname)
setupfolder(folder; remove=nothing)

## Problem data

voltage = 10_000     # V
θr = 293.15          # K
thickness0 = 0.001   # m (1mm)


## Constitutive model

# Thermal model parameters
cv0 = 9.4e5   # Specific heat capacity [J/K/m3]
γv  = 1.0     # Volumetric thermal coupling [-]
κr  = 2.5e9   # Bulk modulus [Pa]
α   = 1.8e-4  # Thermal expansion coefficient [-]
κ   = 0.16    # Thermal conductivity [W/m/K]

# Nonlinear Mooney-Rivlin parameters
μe1 = 4.6e2   # [Pa]
μe2 = 3.8e4   # [Pa]
α1  = 2.0     # [-]
α2  = 1.3     # [-]

# Viscous branches
μ1 = 1.1e4    # [Pa]
τ1 = 10^1.8   # [s]
μ2 = 6.6e3    # [Pa]
τ2 = 10^3.5   # [s]
μ3 = 3.7e4    # [Pa]
τ3 = 10^0.63  # [s]

# Thermo-mechanical coupling
θ∞ = 243.15   # [K]
γ∞ = 0.57     # [-]
θα = 310.0    # [K]
γα = 17.0     # [-]
δα = 0.43     # [-]

# Dielectric properties
ε0 = 8.85e-12 # [F/m]
εr = 4.7      # [-]
θε = 570.0    # [K]
γε = 3.0      # [-]

# coercive_volumetric = VolumetricEnergy(λ=κr)
# hyper_elastic_model = NonlinearMooneyRivlin3D(μ1=μe1, μ2=μe2, α1=α1, α2=α2, λ=0.0)
# branch_1 = ViscousIncompressible(IsochoricNeoHookean3D(μ=μ1), τ=τ1)
# branch_2 = ViscousIncompressible(IsochoricNeoHookean3D(μ=μ2), τ=τ2)
# branch_3 = ViscousIncompressible(IsochoricNeoHookean3D(μ=μ3), τ=τ3)
# visco_model = GeneralizedMaxwell(hyper_elastic_model, branch_1, branch_2, branch_3)
# dielec_model = IdealDielectric(ε=εr*ε0)
# thermal_volumetric = ThermalVolumetric(coercive_volumetric, θr=θr, cv0=cv0, α=α, κ=κ, γ=γv)
# thermo_el = NonlinearMeltingLaw(θr=θr, θM=θ∞, γ=γ∞)
# thermo_vis = NonlinearSofteningLaw(θr=θr, θT=θα, γ=γα, δ=δα)
# thermo_dielec = NonlinearMeltingLaw(θr=θr, θM=θε, γ=γε)
# thermal_dielec = ThermoElectroModel(dielec_model, thermo_dielec)
# model = ThermoElectroMech_Bonet(thermal_volumetric, thermal_dielec, visco_model; el=thermo_el, vis=thermo_vis)

dielec_model = IdealDielectric(ε=εr*ε0)
coercive_volumetric = VolumetricEnergy(λ=κr)
hyper_elastic_model = NeoHookean3D(μ=μe2, λ=0.0)
# hyper_elastic_model = NonlinearMooneyRivlin3D(μ1=μe1, μ2=μe2, α1=α1, α2=α2, λ=0.0)
model = ElectroMechModel(dielec_model, coercive_volumetric + hyper_elastic_model)


## Energy derivatives and kinematics

Ψ, P, _... = model()
F_membrane(λ) = TensorValue(λ, 0, 0, 0, λ, 0, 0, 0, 1/λ^2)
F_membrane(λ1, λ3) = TensorValue(λ1, 0, 0, 0, λ1, 0, 0, 0, λ3)

# F1(λ1, λ3) = TensorValue(λ1, 0, 0, 0, λ1, 0, 0, 0, λ3)
# E1(V) = VectorValue(0, 0, V/thickness)

# Ψm, dΨmdF, dΨmdFF = hyper_elastic_model()
# Ψe, dΨedF, dΨedE, dΨedFF, dΨedFE, dΨedEE = dielec_model()
# E_0 = E0(1000)
# dΨedF(Fp, E_0)

# F_1 = F_membrane(1.1)
# F_1p = F_1*Fp

# dΨmdF(F_1)
# dΨmdF(F_1p) - dΨmdF(Fp)

## Solve at Gauss point

function solve_patch(volts, λp, λ0=[1.0, 1.0])
  F(λ1, λ3) = F_membrane(λ1, λ3)*F_membrane(λp)
  E0(V) = VectorValue(0, 0, V/thickness0)
  res(λ, V) = begin
    P0 = P(F(1.0, 1.0), E0(0))
    Pi = P(F(λ[1], λ[2]), E0(V)) - P0
    return [Pi[1,1], Pi[3,3]]
  end
  prob = NonlinearProblem(res, λ0, volts)
  sol = NonlinearSolve.solve(prob, NewtonRaphson(), abstol=1e-6, maxiters=10)
  return sol.u
end


## Plot

p = plot(xlabel="Radial stretch, λ₁ [-]", ylabel="Voltage [V]")

for λp in [1.0, 1.5, 2.0, 3.0]
  v_values = range(1, voltage; step=10)
  λ_values = accumulate((λn, v) -> solve_patch(v, λp, λn), v_values, init=[1.0, 1.0])
  λ1_values = getindex.(λ_values, 1) .* λp
  replace!(x -> x < 0.1 ? NaN : x, λ1_values)
  plot!(λ1_values, v_values, lw=5, label="λp=$λp")
end

display(p);
