using HyperFEM, HyperFEM.ComputationalModels.CartesianTags
using Gridap, GridapSolvers.NonlinearSolvers
using Gridap.FESpaces, Gridap.Adaptivity, Gridap.CellData
using LineSearches: BackTracking
using MultiAssign
using Plots
using Printf
import Plots:mm
import LinearAlgebra:normalize

≲(a,b) = (a <= b) || (a ≈ b)
≳(a,b) = (a >= b) || (a ≈ b)

pname = stem(@__FILE__)
folder = joinpath(@__DIR__, "results")
outpath = joinpath(folder, pname)
setupfolder(folder; remove=".vtu")

t_end = 3.0
Δt = 0.002
voltage = 3000.0  # V
ffreq = 10  # Hz
long = 0.01  # m
width = 0.005
thick = 0.001
direction = normalize(VectorValue(1, 1, 0))
domain = (0.0, long, 0.0, width, 0.0, thick)
partition = 2 .* (5, 4, 2)
geometry = CartesianDiscreteModel(domain, partition)
labels = get_face_labeling(geometry)
add_tag_from_tags!(labels, "bottom", CartesianTags.faceXY0⁺)
add_tag_from_tags!(labels, "top", CartesianTags.faceXY1⁺)
add_tag_from_tags!(labels, "fixed", CartesianTags.face0YZ⁺)
add_tag_from_tags!(labels, "free-end", CartesianTags.face1YZ⁺)
add_tag_from_vertex_filter!(labels, "mid",  geometry, x -> x[3] ≈ 0.25thick)
add_tag_from_vertex_filter!(labels, "hard", geometry, x -> x[3] ≲ 0.25thick)
add_tag_from_vertex_filter!(labels, "soft", geometry, x -> x[3] ≳ 0.25thick)

# Constitutive model
μ1 = 1.37e4   # Pa
μ2 = 1.50e6   # Pa
κ  = 2.5e9    # Pa
ε0 = 8.85e-12 # [F/m]
εr = 4.7      # [-]
θr = 293.15
soft_elastic = NeoHookean3D(λ=κ, μ=μ1)
isotropic = NeoHookean3D(λ=κ, μ=μ2)
fiber = TransverseIsotropy3D(μ=10μ2, α1=1.0, α2=1.0)
hard_elastic = isotropic + fiber
soft_model = ElectroMechModel(IdealDielectric(ε=εr*ε0), soft_elastic)
hard_model = ElectroMechModel(IdealDielectric(ε=1e-20), hard_elastic)

# Setup integration
order = 2
degree = 2 * order
Ω = Triangulation(geometry)
dΩ = Measure(Ω, degree)

Ω₁ = Interior(geometry, tags="soft")
Ω₂ = Interior(geometry, tags="hard")
dΩ₁ = Measure(Ω₁, degree)
dΩ₂ = Measure(Ω₂, degree)

Γ_face = BoundaryTriangulation(Ω, tags="free-end")
dΓ_face = Measure(Γ_face, degree)

# Dirichlet boundary conditions 
dir_u_tags = ["fixed"]
dir_u_vals = [[0.0, 0.0, 0.0]]
dir_u_func = [t -> 1.0]
dir_u = DirichletBC(dir_u_tags, dir_u_vals, dir_u_func)

func = t -> sin(2π*ffreq*t)
dir_φ_tags = ["top", "hard"]
dir_φ_vals = [0.0, voltage]
dir_φ_func = [func, func]
dir_φ = DirichletBC(dir_φ_tags, dir_φ_vals, dir_φ_func)

dir_bc = MultiFieldBC([dir_u, dir_φ])

# Finite Elements
reffe_u = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
reffe_φ = ReferenceFE(lagrangian, Float64, order)

# Test FE Spaces
Vu = TestFESpace(Ω, reffe_u, dir_u, conformity=:H1)
Vφ = TestFESpace(Ω, reffe_φ, dir_φ, conformity=:H1)

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

N   = interpolate_everywhere(direction, Vu)

# Residual and jacobian
Ψ₁, ∂Ψ₁∂F, ∂Ψ₁∂E, ∂∂Ψ₁∂FF, ∂∂Ψ₁∂EF, ∂∂Ψ₁∂EE = soft_model()
Ψ₂, ∂Ψ₂∂F, ∂Ψ₂∂E, ∂∂Ψ₂∂FF, ∂∂Ψ₂∂EF, ∂∂Ψ₂∂EE = hard_model()
F, H, J = get_Kinematics(Kinematics(Mechano, Solid))
E       = get_Kinematics(Kinematics(Electro, Solid))
Eh      = E∘∇(φh⁺)
Fh      = F∘∇(uh⁺)'
Fh⁻     = F∘∇(uh⁻)'

res(Λ) = ((u, φ), (v, vφ)) -> ∫(∇(v)' ⊙ (∂Ψ₁∂F ∘ (F∘∇(u)', E∘∇(φ)   )))dΩ₁ +
                              ∫(∇(v)' ⊙ (∂Ψ₂∂F ∘ (F∘∇(u)', E∘∇(φ), N)))dΩ₂ +
                              -1.0*∫(∇(vφ) ⋅ (∂Ψ₁∂E ∘ (F∘∇(u)', E∘∇(φ)   )))dΩ₁ +
                              -1.0*∫(∇(vφ) ⋅ (∂Ψ₂∂E ∘ (F∘∇(u)', E∘∇(φ), N)))dΩ₂

res_u(Λ) = (u, v) -> ∫(∇(v)' ⊙ (∂Ψ₁∂F ∘ (F∘∇(u)', Eh   )))dΩ₁ +
                     ∫(∇(v)' ⊙ (∂Ψ₂∂F ∘ (F∘∇(u)', Eh, N)))dΩ₂

res_φ(Λ) = (φ, vφ) -> -1.0*∫(∇(vφ) ⋅ (∂Ψ₁∂E ∘ (Fh, E∘∇(φ)   )))dΩ₁ +
                      -1.0*∫(∇(vφ) ⋅ (∂Ψ₂∂E ∘ (Fh, E∘∇(φ), N)))dΩ₂

jac(Λ) = ((u, φ), (du, dφ), (v, vφ)) -> ∫(∇(v)' ⊙ ((∂∂Ψ₁∂FF ∘ (F∘∇(u)', E∘∇(φ))   ) ⊙ ∇(du)'))dΩ₁ +
                                        ∫(∇(v)' ⊙ ((∂∂Ψ₂∂FF ∘ (F∘∇(u)', E∘∇(φ), N)) ⊙ ∇(du)'))dΩ₂ +
                                        ∫(∇(vφ)' ⋅ ((∂∂Ψ₁∂EE ∘ (F∘∇(u)', E∘∇(φ)   )) ⋅ ∇(dφ)))dΩ₁ +
                                        ∫(∇(vφ)' ⋅ ((∂∂Ψ₂∂EE ∘ (F∘∇(u)', E∘∇(φ), N)) ⋅ ∇(dφ)))dΩ₂ +
                                        -1.0*∫(∇(dφ) ⋅ ((∂∂Ψ₁∂EF ∘ (F∘∇(u)', E∘∇(φ)   )) ⊙ ∇(v)'))dΩ₁ +
                                        -1.0*∫(∇(dφ) ⋅ ((∂∂Ψ₂∂EF ∘ (F∘∇(u)', E∘∇(φ), N)) ⊙ ∇(v)'))dΩ₂ +
                                        -1.0*∫(∇(vφ) ⋅ ((∂∂Ψ₁∂EF ∘ (F∘∇(u)', E∘∇(φ))   ) ⊙ ∇(du)'))dΩ₁ +
                                        -1.0*∫(∇(vφ) ⋅ ((∂∂Ψ₂∂EF ∘ (F∘∇(u)', E∘∇(φ), N)) ⊙ ∇(du)'))dΩ₂

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
@multiassign t, pitch, stroke = Float64[]
function postprocess(pvd, step, time, (uh, φh))
  if step % 1 == 0
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
end

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
p3 = plot(t, Ψtot, labels="Ψ", lcolor=:black, width=2, size=(1500,400), margin=8mm, xlabel="Time [s]", ylabel="Power [W]")
display(p3);
p4 = plot(t, Ψdir, labels="Ψφ,Dir", lcolor=:black, width=2, size=(1500,400), margin=8mm, xlabel="Time [s]", ylabel="Power [W]")
display(p4);
