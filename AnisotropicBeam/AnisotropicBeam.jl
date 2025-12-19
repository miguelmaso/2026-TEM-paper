using HyperFEM, HyperFEM.ComputationalModels.CartesianTags
using Gridap, GridapSolvers.NonlinearSolvers
using Gridap.FESpaces
using Printf

pname = stem(@__FILE__)
folder = joinpath(@__DIR__, "results")
outpath = joinpath(folder, pname)
setupfolder(folder; remove=nothing)

t_end = 1.0
Δt = 0.02

long = 0.1  # m
width = 0.01
thick = 0.001
domain = (0.0, long, 0.0, width, 0.0, thick)
partition = (10, 2, 2)
geometry = CartesianDiscreteModel(domain, partition)
labels = get_face_labeling(geometry)
add_tag_from_tags!(labels, "bottom", CartesianTags.faceZ0)
add_tag_from_tags!(labels, "top", CartesianTags.faceZ1)
add_tag_from_tags!(labels, "fixed", CartesianTags.faceX0)
add_tag_from_vertex_filter!(labels, geometry, "mid", x -> x[3] ≈ 0.5thick)

# Constitutive model
μ  = 1.37e4  # Pa
μ1 = 5.64e4  # Pa
τ1 = 0.82    # s
μ2 = 3.15e4  # Pa
τ2 = 10.7    # s
μ3 = 1.98e4  # Pa
τ3 = 500.0   # s
ϵ = 25e-6    # V/m
isotropic = NeoHookean3D(λ=10μ, μ=μ)
fiber = TransverseIsotropy3D(μ=μ, α1=1.0, α2=1.0)
hyper_elastic = isotropic + fiber
branch_1 = ViscousIncompressible(IncompressibleNeoHookean3D(λ=0.0, μ=μ1), τ=τ1)
branch_2 = ViscousIncompressible(IncompressibleNeoHookean3D(λ=0.0, μ=μ2), τ=τ2)
branch_3 = ViscousIncompressible(IncompressibleNeoHookean3D(λ=0.0, μ=μ2), τ=τ3)
visco_elastic = GeneralizedMaxwell(hyper_elastic, branch_1, branch_2, branch_3) 
electric = IdealDielectric(ε=ϵ)
cons_model = ElectroMechModel(electric, visco_elastic)

# Setup integration
order = 1
degree = 2 * order
Ω = Triangulation(geometry)
dΩ = Measure(Ω, degree)

# Dirichlet boundary conditions 
dir_u_tags = ["fixed"]
dir_u_values = [[0.0, 0.0, 0.0]]
dir_u_timesteps = [Λ -> 1.0]
Du = DirichletBC(dir_u_tags, dir_u_values, dir_u_timesteps)

dir_φ_tags = ["mid", "bottom"]
dir_φ_values = [0.0, 30]
dir_φ_timesteps = [Λ->Λ, Λ->Λ]
Dφ = DirichletBC(dir_φ_tags, dir_φ_values, dir_φ_timesteps)

D_bc = MultiFieldBC([Du, Dφ])

# Finite Elements
reffeu = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
reffeφ = ReferenceFE(lagrangian, Float64, order)

# Test FE Spaces
Vu = TestFESpace(geometry, reffeu, D_bc[1], conformity=:H1)
Vφ = TestFESpace(geometry, reffeφ, D_bc[2], conformity=:H1)

# Trial FE Spaces
Uu  = TrialFESpace(Vu, D_bc[1], 1.0)
Uφ  = TrialFESpace(Vφ, D_bc[2], 1.0)
Uun = TrialFESpace(Vu, D_bc[1], 1.0)

# Multifield FE Spaces
V = MultiFieldFESpace([Vu, Vφ])
U = MultiFieldFESpace([Uu, Uφ])

# FE functions
xh  = FEFunction(U, zero_free_values(U))
uh⁻ = FEFunction(Uun, zero_free_values(Uun))

# residual and jacobian function of load factor
update_time_step!(cons_model, Δt)
_ ,∂Ψu, ∂Ψφ, ∂Ψuu, ∂Ψφu, ∂Ψφφ = cons_model()
F, H, J = get_Kinematics(Kinematics(Mechano, Solid))
E       = get_Kinematics(Kinematics(Electro, Solid))
direction = VectorValue(1, 1, 0)
direction /= norm(direction)
N   = interpolate_everywhere(direction, Vu)
Eh  = E∘∇(xh[2])
Fh  = F∘∇(xh[1])'
Fh⁻ = F∘∇(uh⁻)'
A   = initialize_state(visco_elastic, dΩ)

res(Λ) = ((u, φ), (v, vφ)) -> ∫(∇(v)' ⊙ (∂Ψu ∘ (F∘∇(u)', E∘∇(φ), N, Fh⁻, A...)))dΩ -
                              ∫(∇(vφ) ⋅ (∂Ψφ ∘ (F∘∇(u)', E∘∇(φ), N, Fh⁻, A...)))dΩ

jac(Λ) = ((u, φ), (du, dφ), (v, vφ)) -> ∫(∇(v)' ⊙ ((∂Ψuu ∘ (F∘∇(u)', E∘∇(φ), N, Fh⁻, A...)) ⊙ ∇(du)'))dΩ +
                                        ∫(∇(vφ)' ⋅ ((∂Ψφφ ∘ (F∘∇(u)', E∘∇(φ), N, Fh⁻, A...)) ⋅ ∇(dφ)))dΩ -
                                        ∫(∇(dφ) ⋅ ((∂Ψφu ∘ (F∘∇(u)', E∘∇(φ), N, Fh⁻, A...)) ⊙ ∇(v)'))dΩ -
                                        ∫(∇(vφ) ⋅ ((∂Ψφu ∘ (F∘∇(u)', E∘∇(φ), N, Fh⁻, A...)) ⊙ ∇(du)'))dΩ 

# nonlinear solver
ls = LUSolver()
nls = NewtonSolver(ls; maxiter=20, atol=1.e-10, rtol=1.e-8, verbose=true)
solver = FESolver(nls)

# Postprocessor to save results
function postprocess(pvd, step, time, xh)
  if step % 5 == 0
    uh, φh = xh[1], xh[2]
    pvd[time] = createvtk(Ω, outpath * @sprintf("_%03d", step), cellfields=["u" => uh, "φ" => φh])
  end
end

createpvd(outpath) do pvd
  u⁻ = get_free_dof_values(uh⁻)
  step = 0
  time = 0.0
  postprocess(pvd, step, time, xh)
  while time < t_end
    step += 1
    time += Δt
    printstyled(@sprintf("Step: %i\nTime: %.3f s\n", step, time), color=:green, bold=true)

    TrialFESpace!(Uu, D_bc[1], time)
    TrialFESpace!(Uφ, D_bc[2], time)
    
    op = FEOperator(res(time), jac(time), U, V)
    solve!(xh, solver, op)

    postprocess(pvd, step, time, xh)

    update_state!(cons_model, A, Fh, Eh, N, Fh⁻)
    TrialFESpace!(Uun, D_bc[1], time)
    u⁻ .= get_free_dof_values(xh[1])
  end
end
