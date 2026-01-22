using HyperFEM, HyperFEM.ComputationalModels.CartesianTags
using Gridap, GridapSolvers.NonlinearSolvers
using Gridap.FESpaces, Gridap.Adaptivity, Gridap.CellData
using LineSearches: BackTracking
using MultiAssign
using Plots
using Printf
import Plots:mm

import LinearAlgebra:normalize
normalize(a::Gridap.TensorValues.MultiValue) = a / norm(a)

pname = stem(@__FILE__)
folder = joinpath(@__DIR__, "results")
outpath = joinpath(folder, pname)
setupfolder(folder; remove=".vtu")

t_end = 3.0
Δt = 0.002
voltage = 5e3  # V
ffreq = 10  # Hz
long = 0.01  # m
width = 0.005
thick = 0.001
direction = normalize(VectorValue(1, 1, 0))
domain = (0.0, long, 0.0, width, 0.0, thick)
partition = 3 .* (5, 4, 2)
geometry = CartesianDiscreteModel(domain, partition)
labels = get_face_labeling(geometry)
add_tag_from_tags!(labels, "bottom", CartesianTags.faceZ0)
add_tag_from_tags!(labels, "top", CartesianTags.faceZ1)
add_tag_from_tags!(labels, "fixed", CartesianTags.faceX0)
add_tag_from_tags!(labels, "free-end", CartesianTags.faceX1)
add_tag_from_vertex_filter!(labels, geometry, "mid", x -> x[3] ≈ 0.5thick)

# Constitutive model
μ  = 1.37e4  # Pa
μ1 = 5.64e4  # Pa
τ1 = 0.82    # s
μ2 = 3.15e4  # Pa
τ2 = 10.7    # s
μ3 = 1.98e4  # Pa
τ3 = 500.0   # s
ϵ  = 4.0e-11 # V/m
Cv = 17.385
θr = 293.15
κ  = 10μ + μ1 + μ2 + μ3
α  = 22.33e-5 * κ
γv = 0.5
γd = 0.5
isotropic = NeoHookean3D(λ=10μ, μ=μ)
fiber = TransverseIsotropy3D(μ=10μ, α1=1.0, α2=1.0)
hyper_elastic = isotropic + fiber
branch_1 = ViscousIncompressible(IncompressibleNeoHookean3D(λ=0.0, μ=μ1), τ=τ1)
branch_2 = ViscousIncompressible(IncompressibleNeoHookean3D(λ=0.0, μ=μ2), τ=τ2)
branch_3 = ViscousIncompressible(IncompressibleNeoHookean3D(λ=0.0, μ=μ2), τ=τ3)
visco_elastic = GeneralizedMaxwell(hyper_elastic, branch_1, branch_2, branch_3)
electric_model = IdealDielectric(ε=ϵ)
thermal_model = ThermalModel(Cv=Cv, θr=θr, α=α, κ=κ, γv=γv, γd=γd)
cons_model = ThermoElectroMech_Bonet(thermal_model, electric_model, visco_elastic)

# Setup integration
order = 2
degree = 2 * order
Ω = Triangulation(geometry)
dΩ = Measure(Ω, degree)

Γ_face = BoundaryTriangulation(Ω, tags="free-end")
dΓ_face = Measure(Γ_face, degree)

# Dirichlet boundary conditions 
dir_u_tags = ["fixed"]
dir_u_values = [[0.0, 0.0, 0.0]]
dir_u_timesteps = [t -> 1.0]
dir_u = DirichletBC(dir_u_tags, dir_u_values, dir_u_timesteps)

func = t -> sin(2π*ffreq*t)
dir_φ_tags = ["mid", "bottom"]
dir_φ_values = [0.0, voltage]
dir_φ_timesteps = [func, func]
dir_φ = DirichletBC(dir_φ_tags, dir_φ_values, dir_φ_timesteps)

dir_θ = NothingBC()

# Finite Elements
reffe_u = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
reffe_φ = ReferenceFE(lagrangian, Float64, order)
reffe_θ = ReferenceFE(lagrangian, Float64, order)

# Test FE Spaces
Vu = TestFESpace(geometry, reffe_u, dir_u, conformity=:H1)
Vφ = TestFESpace(geometry, reffe_φ, dir_φ, conformity=:H1)
Vθ = TestFESpace(geometry, reffe_θ, dir_θ, conformity=:H1)

Vφ_dir = DirichletFESpace(Vφ)

println("======================================")
println("Mechanical degrees of freedom : $(Vu.nfree)")
println("Electrical degrees of freedom : $(Vφ.nfree)")
println("Thermal degrees of freedom :    $(Vθ.nfree)")
println("Total degrees of freedom :      $(Vu.nfree+Vφ.nfree+Vθ.nfree)")
println("======================================")

# Trial FE Spaces, FE functions and cell/state variables
Uu  = TrialFESpace(Vu, dir_u)
Uφ  = TrialFESpace(Vφ, dir_φ)
Uθ  = TrialFESpace(Vθ, dir_θ)
uh⁺ = FEFunction(Uu, zero_free_values(Uu))
φh⁺ = FEFunction(Uφ, zero_free_values(Uφ))
θh⁺ = FEFunction(Uθ, θr * ones(Vθ.nfree))

Uu⁻ = TrialFESpace(Vu, dir_u)
Uφ⁻ = TrialFESpace(Vφ, dir_φ)
Uθ⁻ = TrialFESpace(Vθ, dir_θ)
uh⁻ = FEFunction(Uu⁻, zero_free_values(Uu))
φh⁻ = FEFunction(Uφ⁻, zero_free_values(Uφ))
θh⁻ = FEFunction(Uθ⁻, θr * ones(Vθ.nfree))

η⁻  = CellState(0.0, dΩ)
D⁻  = CellState(0.0, dΩ)
A   = initialize_state(cons_model, dΩ)
N   = interpolate_everywhere(direction, Vu)


# Residual and jacobian
update_time_step!(cons_model, Δt)
Ψ, ∂Ψ∂F, ∂Ψ∂E, ∂Ψ∂θ, ∂∂Ψ∂FF, ∂∂Ψ∂EE, ∂∂Ψ∂θθ, ∂∂Ψ∂FE, ∂∂Ψ∂Fθ, ∂∂Ψ∂Eθ = cons_model()
D, ∂D∂θ = Dissipation(cons_model)
κ = cons_model.thermo.κ
η(x...) = -∂Ψ∂θ(x...)
∂η∂θ(x...) = -∂∂Ψ∂θθ(x...)
update_η(_, F, E, θ, N, Fn, A...) = (true, η(F, E, θ, N, Fn, A...))
update_D(_, F, E, θ, N, Fn, A...) = (true, D(F, E, θ, N, Fn, A...))
F, H, J = get_Kinematics(Kinematics(Mechano, Solid))
E       = get_Kinematics(Kinematics(Electro, Solid))
Eh      = E∘∇(φh⁺)
Fh      = F∘∇(uh⁺)'
Fh⁻     = F∘∇(uh⁻)'

res_elec(Λ) = (φ, vφ) -> -1.0*∫(∇(vφ)' ⋅ (∂Ψ∂E ∘ (Fh, E∘(∇(φ)), θh⁺, N, Fh⁻, A...)))dΩ
jac_elec(Λ) = (φ, dφ, vφ) -> ∫(∇(vφ) ⋅ ((∂∂Ψ∂EE ∘ (Fh, E∘(∇(φ)), θh⁺, N, Fh⁻, A...)) ⋅ ∇(dφ)))dΩ

res_mec(Λ) = (u, v) -> ∫(∇(v)' ⊙ (∂Ψ∂F ∘ (F∘(∇(u)'), Eh, θh⁺, N, Fh⁻, A...)))dΩ
jac_mec(Λ) = (u, du, v) -> ∫(∇(v)' ⊙ ((∂∂Ψ∂FF ∘ (F∘(∇(u)'), Eh, θh⁺, N, Fh⁻, A...)) ⊙ (∇(du)')))dΩ

res_therm(Λ) = (θ, vθ) -> begin (
   1/Δt*∫( (θ*(η∘(Fh, Eh, θ, N, Fh⁻, A...)) -θh⁻*η⁻)*vθ )dΩ +
  -1/Δt*0.5*∫( (η∘(Fh, Eh, θ, N, Fh⁻, A...) + η⁻)*(θ - θh⁻)*vθ )dΩ +
  -0.5*∫( (D∘(Fh, Eh, θ, N, Fh⁻, A...) + D⁻)*vθ )dΩ +
   0.5*∫( κ*∇(θ)·∇(vθ) + κ*∇(θh⁻)·∇(vθ) )dΩ
)
end
jac_therm(Λ) = (θ, dθ, vθ) -> begin (
   1/Δt*∫( (η∘(Fh, Eh, θ, N, Fh⁻, A...) + θ*(∂η∂θ∘(Fh, Eh, θ, N, Fh⁻, A...)))*dθ*vθ )dΩ +
  -1/Δt*0.5*∫( (∂η∂θ∘(Fh, Eh, θ, N, Fh⁻, A...)*(θ - θh⁻) + η∘(Fh, Eh, θ, N, Fh⁻, A...) + η⁻)*dθ*vθ )dΩ +
  -0.5*∫( (∂D∂θ∘(Fh, Eh, θ, N, Fh⁻, A...))*dθ*vθ )dΩ +
  ∫( 0.5*κ*∇(dθ)·∇(vθ) )dΩ
)
end

# nonlinear solver
ls = LUSolver()
nls = NewtonSolver(ls; maxiter=10, atol=1.e-9, rtol=1.e-8, verbose=true)
solver = FESolver(nls)

# Postprocessor to save results
geom_out = refine(geometry, order)
Ω_out = Triangulation(geom_out)
reffe_u_out = ReferenceFE(lagrangian, VectorValue{3,Float64}, 1)
reffe_φ_out = ReferenceFE(lagrangian, Float64, 1)
Vu_out = FESpace(geom_out, reffe_u_out)
Vφ_out = FESpace(geom_out, reffe_φ_out)
@multiassign t, pitch, stroke, Ψmec, Ψele, Ψthe, Ψdir, Dvis, ηtot, θavg = Float64[]
function postprocess(pvd, step, time, (uh, φh, θh))
  if step % 5 == 0
    uh_out = interpolate_everywhere(Interpolable(uh), Vu_out)
    φh_out = interpolate_everywhere(Interpolable(φh), Vφ_out)
    pvd[time] = createvtk(Ω_out, outpath * @sprintf("_%03d", step), cellfields=["u" => uh_out, "φ" => φh_out])
  end
  n1 = VectorValue(1, 0, 0)
  n2 = VectorValue(0, 1, 0)
  p = sum(∫( acos ∘ (normalize ∘ (Fh · n2) · n2) )dΓ_face) / sum(∫(1)dΓ_face)
  s = sum(∫( acos ∘ (normalize ∘ (Fh · n1) · n1) )dΓ_face) / sum(∫(1)dΓ_face)
  push!(t, time)
  push!(pitch, p)
  push!(stroke, s)
  b_φ = assemble_vector(vφ -> res_elec(time)(φh⁺, vφ), Vφ_dir)[:]
  ∂φt_fix = (get_dirichlet_dof_values(Uφ) - get_dirichlet_dof_values(Uφ⁻)) / Δt
  θ1h = FEFunction(Vθ, ones(Vθ.nfree))
  push!(Ψmec, sum(res_mec(time)(uh⁺, uh⁺-uh⁻))/Δt)
  push!(Ψele, sum(res_elec(time)(φh⁺, φh⁺-φh⁻))/Δt)
  push!(Ψthe, sum(res_therm(time)(θh⁺, θ1h)))
  push!(Ψdir, b_φ · ∂φt_fix)
  push!(Dvis, sum(∫( D∘(Fh, Eh, θh⁺, N, Fh⁻, A...) )dΩ))
  push!(ηtot, sum(∫( η∘(Fh, Eh, θh⁺, N, Fh⁻, A...) )dΩ))
  push!(θavg, sum(∫( θh⁺ )dΩ) / sum(∫(1)dΩ))
end

update_state!(update_η, η⁻, Fh, Eh, θh⁺, N, Fh⁻, A...)
update_state!(update_D, D⁻, Fh, Eh, θh⁺, N, Fh⁻, A...)
update_state!(cons_model, A, Fh, Eh, θh⁺, N, Fh⁻)

createpvd(outpath) do pvd
  u⁻ = get_free_dof_values(uh⁻)
  φ⁻ = get_free_dof_values(φh⁻)
  θ⁻ = get_free_dof_values(θh⁻)
  step = 0
  time = 0.0
  postprocess(pvd, step, time, (uh⁺, φh⁺, θh⁺))
  while time < t_end
    step += 1
    time += Δt
    printstyled(@sprintf("Step: %i\nTime: %.3f s\n", step, time), color=:green, bold=true)

    TrialFESpace!(Uφ, dir_φ, time)
    TrialFESpace!(Uu, dir_u, time)
    TrialFESpace!(Uθ, dir_θ, time)

    printstyled("Electric step\n", bold=true)
    op_elec = FEOperator(res_elec(time), jac_elec(time), Uφ, Vφ)
    solve!(φh⁺, solver, op_elec)

    printstyled("Mechanical step\n", bold=true)
    op_mec = FEOperator(res_mec(time), jac_mec(time), Uu, Vu)
    solve!(uh⁺, solver, op_mec)

    printstyled("Thermal step\n", bold=true)
    op_therm = FEOperator(res_therm(time), jac_therm(time), Uθ, Vθ)
    solve!(θh⁺, solver, op_therm)

    postprocess(pvd, step, time, (uh⁺, φh⁺, θh⁺))

    update_state!(update_η, η⁻, Fh, Eh, θh⁺, N, Fh⁻, A...)
    update_state!(update_D, D⁻, Fh, Eh, θh⁺, N, Fh⁻, A...)
    update_state!(cons_model, A, Fh, Eh, θh⁺, N, Fh⁻)

    φ⁻ .= get_free_dof_values(φh⁺)
    u⁻ .= get_free_dof_values(uh⁺)
    θ⁻ .= get_free_dof_values(θh⁺)
  end
end

p1 = plot(t, [pitch stroke], labels= ["Pitch" "Stroke"], style=[:dash :solid], lcolor=:black, width=2, size=(1500, 400))
display(p1);
Ψint = Ψmec + Ψele + Ψthe
Ψtot = Ψint - Ψdir
p2 = plot(t, [Ψdir Ψtot Dvis], labels=["Ψφ,Dir" "Ψ" "Dvis"], style=[:dash :solid :dashdot], lcolor=[:black :gray :black], width=2, margin=8mm, xlabel="Time [s]", ylabel="Power [W]")
display(p2);
