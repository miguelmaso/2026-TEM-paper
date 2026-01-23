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

t_end = 0.2
Δt = 0.002
voltage = 5e3  # V
ffreq = 10  # Hz
long = 0.01  # m
width = 0.005
thick = 0.001
direction = normalize(VectorValue(1, 1, 0))
domain = (0.0, long, 0.0, width, 0.0, thick)
partition = 2 .* (5, 4, 2)
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
cons_model = ElectroMechModel(electric_model, visco_elastic)

# Setup integration
order = 2
degree = 2 * order
Ω = Triangulation(geometry)
dΩ = Measure(Ω, degree)

Γ_face = BoundaryTriangulation(Ω, tags="free-end")
dΓ_face = Measure(Γ_face, degree)

# Dirichlet boundary conditions 
dir_u_tags = ["fixed"]
dir_u_vals = [[0.0, 0.0, 0.0]]
dir_u_func = [t -> 1.0]
dir_u = DirichletBC(dir_u_tags, dir_u_vals, dir_u_func)

func = t -> sin(2π*ffreq*t)
dir_φ_tags = ["mid", "bottom"]
dir_φ_vals = [0.0, voltage]
dir_φ_func = [func, func]
dir_φ = DirichletBC(dir_φ_tags, dir_φ_vals, dir_φ_func)

dir_bc = MultiFieldBC([dir_u, dir_φ])

# Finite Elements
reffe_u = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
reffe_φ = ReferenceFE(lagrangian, Float64, order)

# Test FE Spaces
Vu = TestFESpace(geometry, reffe_u, dir_u, conformity=:H1)
Vφ = TestFESpace(geometry, reffe_φ, dir_φ, conformity=:H1)

Vφ_dir = DirichletFESpace(Vφ)

println("======================================")
println("Mechanical degrees of freedom : $(lpad(Vu.nfree,6))")
println("Electrical degrees of freedom : $(lpad(Vφ.nfree,6))")
println("Total degrees of freedom :      $(lpad(Vu.nfree+Vφ.nfree,6))")
println("======================================")

# Trial FE Spaces, FE functions and cell/state variables
Uu  = TrialFESpace(Vu, dir_u)
Uφ  = TrialFESpace(Vφ, dir_φ)
V   = MultiFieldFESpace([Vu, Vφ])
U   = MultiFieldFESpace([Uu, Uφ])
xh  = FEFunction(U, zero_free_values(U))
uh⁺ = xh[1]
φh⁺ = xh[2]

Uu⁻ = TrialFESpace(Vu, dir_u)
Uφ⁻ = TrialFESpace(Vφ, dir_φ)
uh⁻ = FEFunction(Uu⁻, zero_free_values(Uu))
φh⁻ = FEFunction(Uφ⁻, zero_free_values(Uφ))

A   = initialize_state(cons_model, dΩ)
N   = interpolate_everywhere(direction, Vu)

# Residual and jacobian
update_time_step!(cons_model, Δt)
Ψ, ∂Ψ∂F, ∂Ψ∂E, ∂∂Ψ∂FF, ∂∂Ψ∂EF, ∂∂Ψ∂EE = cons_model()
D = Dissipation(cons_model)
F, H, J = get_Kinematics(Kinematics(Mechano, Solid))
E       = get_Kinematics(Kinematics(Electro, Solid))
Eh      = E∘∇(φh⁺)
Fh      = F∘∇(uh⁺)'
Fh⁻     = F∘∇(uh⁻)'

res(Λ) = ((u, φ), (v, vφ)) -> ∫(∇(v)' ⊙ (∂Ψ∂F ∘ (F∘∇(u)', E∘∇(φ), N, Fh⁻, A...)))dΩ -
                              ∫(∇(vφ) ⋅ (∂Ψ∂E ∘ (F∘∇(u)', E∘∇(φ), N, Fh⁻, A...)))dΩ

res_u(Λ) = (u, v) -> ∫(∇(v)' ⊙ (∂Ψ∂F ∘ (F∘∇(u)', Eh, N, Fh⁻, A...)))dΩ

res_φ(Λ) = (φ, vφ) -> -1.0*∫(∇(vφ) ⋅ (∂Ψ∂E ∘ (Fh, E∘∇(φ), N, Fh⁻, A...)))dΩ

jac(Λ) = ((u, φ), (du, dφ), (v, vφ)) -> ∫(∇(v)' ⊙ ((∂∂Ψ∂FF ∘ (F∘∇(u)', E∘∇(φ), N, Fh⁻, A...)) ⊙ ∇(du)'))dΩ +
                                        ∫(∇(vφ)' ⋅ ((∂∂Ψ∂EE ∘ (F∘∇(u)', E∘∇(φ), N, Fh⁻, A...)) ⋅ ∇(dφ)))dΩ -
                                        ∫(∇(dφ) ⋅ ((∂∂Ψ∂EF ∘ (F∘∇(u)', E∘∇(φ), N, Fh⁻, A...)) ⊙ ∇(v)'))dΩ -
                                        ∫(∇(vφ) ⋅ ((∂∂Ψ∂EF ∘ (F∘∇(u)', E∘∇(φ), N, Fh⁻, A...)) ⊙ ∇(du)'))dΩ 

# nonlinear solver
ls = LUSolver()
nls = NewtonSolver(ls; maxiter=10, atol=1.e-10, rtol=1.e-10, verbose=true)
solver = FESolver(nls)

# Postprocessor to save results
geom_out = refine(geometry, order)
Ω_out = Triangulation(geom_out)
reffe_u_out = ReferenceFE(lagrangian, VectorValue{3,Float64}, 1)
reffe_φ_out = ReferenceFE(lagrangian, Float64, 1)
Vu_out = FESpace(geom_out, reffe_u_out)
Vφ_out = FESpace(geom_out, reffe_φ_out)
@multiassign t, pitch, stroke, Ψmec, Ψele, Ψdir, Dvis = Float64[]
function postprocess(pvd, step, time, (uh, φh))
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
  bφ_dir = assemble_vector(v -> res_φ(time)(xh[2], v), Vφ_dir)[:]
  ∂φt_fix = (get_dirichlet_dof_values(Uφ) - get_dirichlet_dof_values(Uφ⁻)) / Δt
  push!(Ψmec, sum(res_u(time)(uh, uh-uh⁻))/Δt)
  push!(Ψele, sum(res_φ(time)(φh, φh-φh⁻))/Δt)
  push!(Ψdir, bφ_dir · ∂φt_fix)
  push!(Dvis, sum(∫( D∘(Fh, Eh, N, Fh⁻, A...) )dΩ))
end

update_state!(cons_model, A, Fh, Eh, N, Fh⁻)

createpvd(outpath) do pvd
  u⁻ = get_free_dof_values(uh⁻)
  φ⁻ = get_free_dof_values(φh⁻)
  step = 0
  time = 0.0
  postprocess(pvd, step, time, xh)
  while time < t_end
    step += 1
    time += Δt
    printstyled(@sprintf("Step: %i\nTime: %.3f s\n", step, time), color=:green, bold=true)

    TrialFESpace!(Uφ, dir_φ, time)
    TrialFESpace!(Uu, dir_u, time)

    op = FEOperator(res(time), jac(time), U, V)
    solve!(xh, solver, op)

    postprocess(pvd, step, time, xh)

    update_state!(cons_model, A, Fh, Eh, N, Fh⁻)
    TrialFESpace!(Uφ⁻, dir_φ, time)
    TrialFESpace!(Uu⁻, dir_u, time)
    u⁻ .= get_free_dof_values(xh[1])
    φ⁻ .= get_free_dof_values(xh[2])
  end
end

p1 = plot(t, (180/π).*[pitch stroke], labels= ["Pitch" "Stroke"], style=[:solid :solid], lcolor=[:gray :black], width=2, size=(1500, 400), margin=8mm, xlabel="Time [s]", ylabel="Angle [º]")
display(p1);
Ψint = Ψmec + Ψele
Ψtot = Ψint - Ψdir
# p2 = plot(t, [Ψdir Ψtot Dvis], labels=["Ψφ,Dir" "Ψ" "Dvis"], style=[:dash :solid :dashdot], lcolor=[:black :gray :black], width=2, margin=8mm, xlabel="Time [s]", ylabel="Power [W]")
# p2 = plot(t, [Ψtot Dvis], labels=["Ψ" "Dvis"], style=[:solid :dashdot], lcolor=[:gray :black], width=2, margin=8mm, xlabel="Time [s]", ylabel="Power [W]")
p2 = plot(t, Dvis, labels="Dvis", lcolor=:black, width=2, size=(1500,400), margin=8mm, xlabel="Time [s]", ylabel="Power [W]")
display(p2);
