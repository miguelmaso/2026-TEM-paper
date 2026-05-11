using HyperFEM
using HyperFEM.ComputationalModels.PostMetrics
using HyperFEM.ComputationalModels.CartesianTags
using HyperFEM.ComputationalModels.EvolutionFunctions
using Gridap, Gridap.FESpaces
using GridapSolvers, GridapSolvers.NonlinearSolvers
using Printf
using Plots
using MultiAssign
import Plots:mm

pname = stem(@__FILE__)
folder = joinpath(@__DIR__, "results")
outpath = joinpath(folder, pname)
setupfolder(folder; remove=".vtu")

## Domain

size = 0.1   # 10cm
thick = 0.001 # 1mm
ndivisions = 4
domain = (-0.5*size, 0.5*size, -0.5*size, 0.5*size, 0.0, thick)
partition = (2*ndivisions, 2*ndivisions, ndivisions)
geometry = CartesianDiscreteModel(domain, partition)
labels = get_face_labeling(geometry)
add_tag_from_tags!(labels, "top", CartesianTags.faceZ1)
add_tag_from_tags!(labels, "bottom", CartesianTags.faceZ0)
add_tag_from_tags!(labels, "edge", CartesianTags.edgeX00)
add_tag_from_tags!(labels, "corner", CartesianTags.corner000)
add_tag_from_tags!(labels, "faces", [CartesianTags.faceX0; CartesianTags.faceX1; CartesianTags.faceY0; CartesianTags.faceY1])
add_tag_from_vertex_filter!(labels, geometry, "top_electrode", p -> p[3] ≈ thick && sqrt(p[1]^2 + p[2]^2) < 0.25*size+1e-6)
add_tag_from_vertex_filter!(labels, geometry, "bottom_electrode", p -> p[3] ≈ 0.0   && sqrt(p[1]^2 + p[2]^2) < 0.25*size+1e-6)

writevtk(geometry, outpath*"_geom")

## Constitutive model

# Thermal model parameters
θr  = 293.15   # Reference temperature [K]
cv0 = 9.4e5    # Specific heat capacity [J/K/m3]
γv  = 1.0      # Volumetric thermal coupling [-]
κr  = 2.5e9    # Bulk modulus [Pa]
α   = 1.8e-4   # Thermal expansion coefficient [-]
κ   = 0.16     # Thermal conductivity [W/m/K]

# Nonlinear Mooney-Rivlin parameters
μe1 = 4.6e2  # [Pa]
μe2 = 3.8e4  # [Pa]
α1  = 2.0    # [-]
α2  = 1.3    # [-]

# Viscous branches
μ1 = 1.1e4    # [Pa]
τ1 = 10^1.8   # [s]
μ2 = 6.6e3    # [Pa]
τ2 = 10^3.5   # [s]
μ3 = 3.7e4    # [Pa]
τ3 = 10^0.63  # [s]

# Thermo-mechanical coupling
θ∞ = 243.15 # [K]
γ∞ = 0.57   # [-]
θα = 310.0  # [K]
γα = 17.0   # [-]
δα = 0.43   # [-]

# Dielectric properties
ε0 = 8.85e-12 # [F/m]
ε  = 4.7      # [-]
θε = 570.0    # [K]
γε = 3.0      # [-]

coercive_volumetric = VolumetricEnergy(λ=κr)
hyper_elastic_model = NonlinearMooneyRivlin3D(μ1=μe1, μ2=μe2, α1=α1, α2=α2, λ=0.0)
branch_1 = ViscousIncompressible(IsochoricNeoHookean3D(μ=μ1), τ=τ1)
branch_2 = ViscousIncompressible(IsochoricNeoHookean3D(μ=μ2), τ=τ2)
branch_3 = ViscousIncompressible(IsochoricNeoHookean3D(μ=μ3), τ=τ3)
visco_model = GeneralizedMaxwell(hyper_elastic_model, branch_1, branch_2, branch_3)
dielec_model = IdealDielectric(ε=ε*ε0)
thermal_volumetric = ThermalVolumetric(coercive_volumetric, θr=θr, cv0=cv0, α=α, κ=κ, γ=γv)
thermo_el = NonlinearMeltingLaw(θr=θr, θM=θ∞, γ=γ∞)
thermo_vis = NonlinearSofteningLaw(θr=θr, θT=θα, γ=γα, δ=δα)
thermo_dielec = NonlinearMeltingLaw(θr=θr, θM=θε, γ=γε)
thermal_dielec = ThermoElectroModel(dielec_model, thermo_dielec)
model = ThermoElectroMech_Bonet(thermal_volumetric, thermal_dielec, visco_model; el=thermo_el, vis=thermo_vis)

## Kinematics

struct PreStrech end

function HyperFEM.get_Kinematics(::Type{PreStrech})
  λp = 1.5
  Fp = TensorValue{3,3}(λp, 0.0, 0.0, 0.0, λp, 0.0, 0.0, 0.0, λp^(-2))
  F(∇u) = Fp + ∇u
  H(F) = cof(F)
  J(F) = det(F)
  return F, H, J
end

ku = PreStrech
ke = Kinematics(Electro, Solid)
kt = Kinematics(Thermo, Solid)
F, H, J = get_Kinematics(ku)
E       = get_Kinematics(ke)

## Discrete domain, integration and boundary conditions

order = 2
degree = 2 * order
Ω = Triangulation(geometry)
dΩ = Measure(Ω, degree)
t_end = 2.0  # s
Δt = 0.02    # s
update_time_step!(model, Δt)

solver_mech = FESolver(NewtonSolver(LUSolver(); maxiter=20, atol=1e-8, rtol=1e-8, verbose=true))
solver_elec = FESolver(NewtonSolver(LUSolver(); maxiter=20, atol=1e-10, rtol=1e-10, verbose=true))
solver_therm = FESolver(NewtonSolver(LUSolver(); maxiter=20, atol=1e-10, rtol=1e-10, verbose=true))

dir_u_tags = ["faces"]
dir_u_values = [[0.0, 0.0, 0.0]]
dir_u_time = [Λ->1]
dir_u_masks = [[true,true,true]]
dirichlet_u = DirichletBC(dir_u_tags, dir_u_values, dir_u_time)

voltage = 1000.0
dir_φ_tags = ["top_electrode", "bottom"]
dir_φ_values = [voltage, 0.0]
dir_φ_time = [ramp(1.0), Λ->1]
dirichlet_φ = DirichletBC(dir_φ_tags, dir_φ_values, dir_φ_time)

dirichlet_θ = NothingBC()

reffeu = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
reffeφ = ReferenceFE(lagrangian, Float64, order)
reffeθ = ReferenceFE(lagrangian, Float64, order)

Vu = TestFESpace(geometry, reffeu, dirichlet_u, conformity=:H1, dirichlet_masks=dir_u_masks)
Vφ = TestFESpace(geometry, reffeφ, dirichlet_φ, conformity=:H1)
Vθ = TestFESpace(geometry, reffeθ, dirichlet_θ, conformity=:H1)

Vφ_dir = DirichletFESpace(Vφ)

println("======================================")
println("Mechanical degrees of freedom : $(Vu.nfree)")
println("Electrical degrees of freedom : $(Vφ.nfree)")
println("Thermal degrees of freedom :    $(Vθ.nfree)")
println("Total degrees of freedom :      $(Vu.nfree+Vφ.nfree+Vθ.nfree)")
println("======================================")

## Trial FE spaces and state variables

Uu  = TrialFESpace(Vu, dirichlet_u)
Uφ  = TrialFESpace(Vφ, dirichlet_φ)
Uθ  = TrialFESpace(Vθ, dirichlet_θ)
uh⁺ = FEFunction(Uu, zero_free_values(Uu))
φh⁺ = FEFunction(Uφ, zero_free_values(Uφ))
θh⁺ = FEFunction(Uθ, θr * ones(Vθ.nfree))

Uu⁻ = TrialFESpace(Vu, dirichlet_u)
Uφ⁻ = TrialFESpace(Vφ, dirichlet_φ)
Uθ⁻ = TrialFESpace(Vθ, dirichlet_θ)
uh⁻ = FEFunction(Uu⁻, zero_free_values(Uu))
φh⁻ = FEFunction(Uφ⁻, zero_free_values(Uφ))
θh⁻ = FEFunction(Uθ⁻, θr * ones(Vθ.nfree))

η⁻  = CellState(0.0, dΩ)
D⁻  = CellState(0.0, dΩ)

Eh  = E∘∇(φh⁺)
Eh⁻ = E∘∇(φh⁻)
Fh  = F∘∇(uh⁺)'
Fh⁻ = F∘∇(uh⁻)'
A   = initialize_state(visco_model, dΩ)

## Weak forms: residual and jacobian

Ψ, ∂Ψ∂F, ∂Ψ∂E, ∂Ψ∂θ, ∂∂Ψ∂FF, ∂∂Ψ∂EE, ∂∂Ψ∂θθ, ∂∂Ψ∂FE, ∂∂Ψ∂Fθ, ∂∂Ψ∂Eθ = model()
D, ∂D∂θ = Dissipation(model)
η(x...) = -∂Ψ∂θ(x...)
∂η∂θ(x...) = -∂∂Ψ∂θθ(x...)
update_η(_, θ, E, F, Fn, A...) = (true, η(F, E, θ, Fn, A...))
update_D(_, θ, E, F, Fn, A...) = (true, D(F, E, θ, Fn, A...))
κ = model.thermo.thermo.κ

# Electro
res_elec(Λ) = (φ, vφ) -> -1.0*∫(∇(vφ)' ⋅ (∂Ψ∂E ∘ (F∘(∇(uh⁺)'), E∘(∇(φ)), θh⁺, Fh⁻, A...)))dΩ
jac_elec(Λ) = (φ, dφ, vφ) -> ∫(∇(vφ) ⋅ ((∂∂Ψ∂EE ∘ (F∘(∇(uh⁺)'), E∘(∇(φ)), θh⁺, Fh⁻, A...)) ⋅ ∇(dφ)))dΩ

# Mechano
res_mec(Λ) = (u, v) -> ∫(∇(v)' ⊙ (∂Ψ∂F ∘ (F∘(∇(u)'), E∘(∇(φh⁺)), θh⁺, Fh⁻, A...)))dΩ
jac_mec(Λ) = (u, du, v) -> ∫(∇(v)' ⊙ ((∂∂Ψ∂FF ∘ (F∘(∇(u)'), E∘(∇(φh⁺)), θh⁺, Fh⁻, A...)) ⊙ (∇(du)')))dΩ

# Thermo
res_therm(Λ) = (θ, vθ) -> begin (
   1/Δt*∫( (θ*(η∘(F∘∇(uh⁺)', E∘∇(φh⁺), θ, Fh⁻, A...)) -θh⁻*η⁻)*vθ )dΩ +
  -1/Δt*0.5*∫( (η∘(F∘∇(uh⁺)', E∘∇(φh⁺), θ, Fh⁻, A...) + η⁻)*(θ - θh⁻)*vθ )dΩ +
  -0.5*∫( (D∘(F∘∇(uh⁺)', E∘∇(φh⁺), θ, Fh⁻, A...) + D⁻)*vθ )dΩ +
   0.5*∫( κ*∇(θ)·∇(vθ) + κ*∇(θh⁻)·∇(vθ) )dΩ
)
end
jac_therm(Λ) = (θ, dθ, vθ) -> begin (
   1/Δt*∫( (η∘(F∘∇(uh⁺)', E∘∇(φh⁺), θ, Fh⁻, A...) + θ*(∂η∂θ∘(F∘∇(uh⁺)', E∘∇(φh⁺), θ, Fh⁻, A...)))*dθ*vθ )dΩ +
  -1/Δt*0.5*∫( (∂η∂θ∘(F∘∇(uh⁺)', E∘∇(φh⁺), θ, Fh⁻, A...)*(θ - θh⁻) + η∘(F∘∇(uh⁺)', E∘∇(φh⁺), θ, Fh⁻, A...) + η⁻)*dθ*vθ )dΩ +
  -0.5*∫( (∂D∂θ∘(F∘∇(uh⁺)', E∘∇(φh⁺), θ, Fh⁻, A...))*dθ*vθ )dΩ +
  ∫( 0.5*κ*∇(dθ)·∇(vθ) )dΩ
)
end

## Post-processor

@multiassign Ψmec, Ψele, Ψthe, Ψdir, Dvis, ηtot, θavg, umax, ∂Pθ_F, ∂Dθ_E, cv = Float64[]
function driverpost(pvd, step, time)
  b_φ = assemble_vector(vφ -> res_elec(time)(φh⁺, vφ), Vφ_dir)[:]
  ∂φt_fix = (get_dirichlet_dof_values(Uφ) - get_dirichlet_dof_values(Uφ⁻)) / Δt
  θ1h = FEFunction(Vθ, ones(Vθ.nfree))
  push!(Ψmec, sum(res_mec(time)(uh⁺, uh⁺-uh⁻))/Δt)
  push!(Ψele, sum(res_elec(time)(φh⁺, φh⁺-φh⁻))/Δt)
  push!(Ψthe, sum(res_therm(time)(θh⁺, θ1h)))
  push!(Ψdir, b_φ · ∂φt_fix)
  push!(Dvis, sum(∫( D∘(Fh, Eh, θh⁺, Fh⁻, A...) )dΩ))
  push!(ηtot, sum(∫( η∘(Fh, Eh, θh⁺, Fh⁻, A...) )dΩ))
  push!(θavg, sum(∫( θh⁺ )dΩ) / sum(∫(1)dΩ))
  push!(umax, component_LInf(uh⁺, :z, Ω))
  push!(∂Pθ_F, sum(∫( (∂∂Ψ∂Fθ∘(Fh, Eh, θh⁺, Fh⁻, A...))⊙(Fh-Fh⁻)/Δt )dΩ))
  push!(∂Dθ_E, sum(∫( -(∂∂Ψ∂Eθ∘(Fh, Eh, θh⁺, Fh⁻, A...))⋅(Eh-Eh⁻)/Δt )dΩ))
  push!(cv,    sum(∫( -(∂∂Ψ∂θθ∘(Fh, Eh, θh⁺, Fh⁻, A...)) )dΩ))
  if mod(step, 5) == 0
    ηh = interpolate_L2_scalar(η∘(Fh, Eh, θh⁺, Fh⁻, A...), Ω, dΩ)
    pvd[time] = createvtk(Ω, outpath * @sprintf("_%03d", step), cellfields=["u" => uh⁺, "ϕ" => φh⁺, "θ" => θh⁺, "η" => ηh])
  end
end

## Time integration

update_state!(update_η, η⁻, θh⁺, Eh, Fh, Fh⁻, A...)
update_state!(update_D, D⁻, θh⁺, Eh, Fh, Fh⁻, A...)

createpvd(outpath) do pvd
  u⁻ = get_free_dof_values(uh⁻)
  φ⁻ = get_free_dof_values(φh⁻)
  θ⁻ = get_free_dof_values(θh⁻)
  step = 0
  time = 0
  driverpost(pvd, step, time)
  println("Entering the time loop")
  while time < t_end
    step += 1
    time += Δt
    printstyled(@sprintf("Step: %i\nTime: %.3f s\n", step, time), color=:green, bold=true)

    #-----------------------------------------
    # Update boundary conditions
    #-----------------------------------------
    TrialFESpace!(Uφ, dirichlet_φ, time)
    TrialFESpace!(Uu, dirichlet_u, time)
    TrialFESpace!(Uθ, dirichlet_θ, time)

    println("Electric staggered step")
    op_elec = FEOperator(res_elec(time), jac_elec(time), Uφ, Vφ)
    solve!(φh⁺, solver_elec, op_elec)

    println("Mechanical staggered step")
    op_mech = FEOperator(res_mec(time), jac_mec(time), Uu, Vu)
    solve!(uh⁺, solver_mech, op_mech)

    println("Thermal staggered step")
    op_therm = FEOperator(res_therm(time), jac_therm(time), Uθ, Vθ)
    solve!(θh⁺, solver_therm, op_therm)

    #-----------------------------------------
    # Post processing
    #-----------------------------------------
    driverpost(pvd, step, time)

    #-----------------------------------------
    # Update boundary conditions and old step
    #-----------------------------------------
    update_state!(update_η, η⁻, θh⁺, Eh, Fh, Fh⁻, A...)
    update_state!(update_D, D⁻, θh⁺, Eh, Fh, Fh⁻, A...)
    update_state!(visco_model, A, Fh, Fh⁻)

    TrialFESpace!(Uφ⁻, dirichlet_φ, time)
    TrialFESpace!(Uu⁻, dirichlet_u, time)
    TrialFESpace!(Uθ⁻, dirichlet_θ, time)

    φ⁻ .= get_free_dof_values(φh⁺)
    u⁻ .= get_free_dof_values(uh⁺)
    θ⁻ .= get_free_dof_values(θh⁺)
  end
end

## Metrics visualization and check

η_ref = ηtot[1]
times = [0:Δt:t_end]
p1 = plot(times, ηtot, labels="Entropy", style=:solid, lcolor=:black, width=2, ylim=[1-5.1e-3, 1+5.1e-3]*η_ref, yticks=[1-5e-3, 1, 1+5e-3]*η_ref, margin=8mm, xlabel="Time [s]", ylabel="Entropy [J/K]")
p1 = plot!(p1, times, NaN.*times, labels="Temperature", style=:dash, lcolor=:gray, width=2)
p1 = plot!(twinx(p1), times, θavg, labels="Temperature", style=:dash, lcolor=:gray, width=2, xticks=false, legend=false, ylabel="Temperature [ºK]")
Ψint = Ψmec + Ψele + Ψthe
Ψtot = Ψint - Ψdir
p2 = plot(times, [Ψint Ψdir Ψtot Dvis], labels=["Ψu+Ψφ+Ψθ" "Ψφ,Dir" "Ψ" "Dvis"], style=[:solid :dash :solid :dashdot], lcolor=[:black :black :gray :black], width=2, margin=8mm, xlabel="Time [s]", ylabel="Power [W]")
p3 = plot(times, umax, labels="uz,L∞", color=:black, width=2, margin=8mm, xlabel="Time [s]", ylabel="Displacement [m]")
p4 = plot(p1, p2, p3, layout=@layout([a b c]), size=(1200, 500))
display(p4);


F1 = TensorValue(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
E0 = VectorValue(zeros(3))
A1 = VectorValue(F1..., 0.0)

Ψv, ∂Ψv∂F, ∂Ψv∂FF = visco_model()
@show (Ψv(F1, F1, A1) / θr - cv0) * 1e-3

trapz(a::AbstractArray) = sum(a) -0.5(a[1] + a[end])

Dvis_θ = Dvis ./ θavg
Dvis_int = trapz(Dvis_θ) * Δt
@show ηtot[end] - ηtot[1]
@show ηtot[end] - ηtot[1] - Dvis_int

@show trapz(Dvis_θ ./ cv)
@show trapz(∂Pθ_F ./ cv)
@show trapz(∂Dθ_E ./ cv)
