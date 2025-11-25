using HyperFEM
using HyperFEM.ComputationalModels.PostMetrics
using HyperFEM.ComputationalModels.CartesianTags
using Gridap, GridapSolvers
using GridapSolvers.NonlinearSolvers
using Gridap.FESpaces
using Printf
using Plots


# pname = stem(@__FILE__)
pname = splitext(basename(@__FILE__))[1]
folder = joinpath(@__DIR__, "results")
setupfolder(folder; remove=".vtu")

size = 0.1  # m
domain = (0.0, size, 0.0, size, 0.0, size)
partition = (3, 3, 3)
geometry = CartesianDiscreteModel(domain, partition)
labels = get_face_labeling(geometry)
add_tag_from_tags!(labels, "bottom", CartesianTags.faceZ0)
add_tag_from_tags!(labels, "top", CartesianTags.faceZ1)
add_tag_from_tags!(labels, "edge", CartesianTags.edgeX00)
add_tag_from_tags!(labels, "corner", CartesianTags.corner000)
writevtk(geometry, folder * "\\geometry")

# Constitutive model parameters
ε  = 1.0
μ1 = 1.0
μ2 = 1.0
λ  = 10.0
Cv = 17.385
θr = 293.15
κ  = λ + 2(μ1+μ2)
α  = 22.33e-5 * κ
γv = 1.0
γd = 1.0

# Constitutive model
hyper_elastic_model = NeoHookean3D(λ=λ, μ=μ1)
elec_model = IdealDielectric(ε=ε)
therm_model = ThermalModel(Cv=Cv, θr=θr, α=α, κ=κ, γv=γv, γd=γd)
cons_model = ThermoElectroMech_Bonet(therm_model, elec_model, hyper_elastic_model)
ku = Kinematics(Mechano, Solid)
ke = Kinematics(Electro, Solid)
kt = Kinematics(Thermo, Solid)
F, _... = get_Kinematics(ku)
E       = get_Kinematics(ke)

# Setup integration
order = 2
degree = 2 * order
Ω = Triangulation(geometry)
dΩ = Measure(Ω, degree)
t_end = 1.0  # s
Δt = 0.02    # s

# Dirichlet boundary conditions 
dir_u_tags = ["corner", "edge", "bottom"]  # The first tag will overwrite the last one.
dir_u_values = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
dir_u_timesteps = [Λ->1, Λ->1, Λ->1]
dir_u_masks = [[true,true,true],[false,true,true],[false,false,true]]
dirichlet_u = DirichletBC(dir_u_tags, dir_u_values, dir_u_timesteps)

voltage = 0.065
dir_φ_tags = ["bottom", "top"]
dir_φ_values = [0.0, voltage]
dir_φ_timesteps = [Λ->1, Λ->Λ]
dirichlet_φ = DirichletBC(dir_φ_tags, dir_φ_values, dir_φ_timesteps)

dirichlet_θ = NothingBC()


# Finite Elements
reffeu = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
reffeφ = ReferenceFE(lagrangian, Float64, order)
reffeθ = ReferenceFE(lagrangian, Float64, order)

# Test FE Spaces
Vu = TestFESpace(geometry, reffeu, dirichlet_u, conformity=:H1, dirichlet_masks=dir_u_masks)
Vφ = TestFESpace(geometry, reffeφ, dirichlet_φ, conformity=:H1)
Vθ = TestFESpace(geometry, reffeθ, dirichlet_θ, conformity=:H1)

Vφ_dir = DirichletFESpace(Vφ)

# Trial FE Spaces and state variables
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

# Previous time step values
u⁻  = get_free_dof_values(uh⁻)
φ⁻  = get_free_dof_values(φh⁻)
θ⁻  = get_free_dof_values(θh⁻)
η⁻  = CellState(0.0, dΩ)

Eh = E∘∇(φh⁺)  # Cuando el solver funcione, hay que ver si estos shortcuts funcionan
Fh = F∘∇(uh⁺)'
Fh⁻ = F∘∇(uh⁻)'
A = initializeStateVariables(cons_model, dΩ)

# =================================
# Weak forms: residual and jacobian
# =================================

Ψ, ∂Ψ∂F, ∂Ψ∂E, ∂Ψ∂θ, ∂∂Ψ∂FF, ∂∂Ψ∂EE, ∂∂Ψ∂θθ, ∂∂Ψ∂FE, ∂∂Ψ∂Fθ, ∂∂Ψ∂Eθ = cons_model()
D, ∂D∂θ = Dissipation(cons_model, Δt)
η(x...) = -∂Ψ∂θ(x...)
∂η∂θ(x...) = -∂∂Ψ∂θθ(x...)
κ = cons_model.thermo.κ

# Electro
res_elec(Λ) = (φ, vφ) -> residual(cons_model, Electro, (ku, ke, kt), (uh⁺, φ, θh⁺), vφ, dΩ)
jac_elec(Λ) = (φ, dφ, vφ) -> jacobian(cons_model, Electro, (ku, ke, kt), (uh⁺, φ, θh⁺), dφ, vφ, dΩ)

# Mechano
res_mec(Λ) = (u, v) -> residual(cons_model, Mechano, (ku, ke, kt), (u, φh⁺, θh⁺), v, dΩ)
jac_mec(Λ) = (u, du, v) -> jacobian(cons_model, Mechano, (ku, ke, kt), (u, φh⁺, θh⁺), du, v, dΩ)

# Thermo
res_therm(Λ) = (θ, vθ) -> begin (
   1/Δt*∫( (θ*(η∘(F∘∇(uh⁺)', E∘∇(φh⁺), θ)) -θh⁻*η⁻)*vθ )dΩ +
  -1/Δt*0.5*∫( (η∘(F∘∇(uh⁺)', E∘∇(φh⁺), θ) + η⁻)*(θ - θh⁻)*vθ )dΩ +
  # -0.5*(D∘(F∘∇(uhᵞ)', E∘∇(φhᵞ), θ) + Dh⁻)*vθ +
   0.5*∫( κ*∇(θ)·∇(vθ) + κ*∇(θh⁻)·∇(vθ) )dΩ
)
end
jac_therm(Λ) = (θ, dθ, vθ) -> begin (
   1/Δt*∫( (η∘(F∘∇(uh⁺)', E∘∇(φh⁺), θ) + θ*(∂η∂θ∘(F∘∇(uh⁺)', E∘∇(φh⁺), θ)))*dθ*vθ )dΩ +
  -1/Δt*0.5*∫( (∂η∂θ∘(F∘∇(uh⁺)', E∘∇(φh⁺), θ)*(θ - θh⁻) + η∘(F∘∇(uh⁺)', E∘∇(φh⁺), θ) + η⁻)*dθ*vθ )dΩ +
  # -0.5*(∂D∂θ∘(F∘∇(uh⁺)', E∘∇(φh⁺), θ))*dθ*vθ +
  ∫( 0.5*κ*∇(dθ)·∇(vθ) )dΩ
)
end
# res_therm_neu(Λ) = vθ -> 0.5*∫( Q(Λ)*vθ + Q(Λ-Δt)*vθ )dΓₙ
# res_therm_tot(Λ) = (θ, vθ) -> res_therm(Λ)(θ, vθ) - res_therm_neu(Λ)(vθ)


ls = LUSolver()
nls = NewtonSolver(ls; maxiter=20, atol=1e-8, rtol=1e-8, verbose=true)
solver = FESolver(nls)

# Postprocessor to save results
Ψmec = Float64[]
Ψele = Float64[]
Ψthe = Float64[]
Ψdir = Float64[]
ηtot = Float64[]
θavg = Float64[]
umax = Float64[]
function driverpost(pvd, step, time)
  b_φ = assemble_vector(vφ -> res_elec(time)(φh⁺, vφ), Vφ_dir)[:]
  ∂φt_fix = (get_dirichlet_dof_values(Uφ) - get_dirichlet_dof_values(Uφ⁻)) / Δt
  θ1_free = ones(Vθ.nfree)
  θ1h = FEFunction(Vθ, θ1_free)
  ηΩ = sum(∫(η∘(F∘∇(uh⁺)', E∘∇(φh⁺), θh⁺))dΩ)
  θΩ = sum(∫(θh⁺)dΩ) / sum(∫(1)dΩ)
  push!(Ψmec, sum(res_mec(time)(uh⁺, uh⁺-uh⁻))/Δt)
  push!(Ψele, sum(res_elec(time)(φh⁺, φh⁺-φh⁻))/Δt)
  push!(Ψthe, sum(res_therm(time)(θh⁺, θ1h)))
  push!(Ψdir, b_φ · ∂φt_fix)
  push!(ηtot, ηΩ)
  push!(θavg, θΩ)
  push!(umax, component_LInf(uh⁺, :z, Ω))
  if mod(step, 1) == 0
    pvd[time] = createvtk(Ω, folder * "/STEP_$step" * ".vtu", cellfields=["u" => uh⁺, "ϕ" => φh⁺, "θ" => θh⁺, "η" => η∘(F∘∇(uh⁺)', E∘∇(φh⁺), θh⁺)])
  end
end


update_η(_, θ, E, F) = (true, η(F, E, θ))
update_state!(update_η, η⁻, θh⁺, E∘∇(φh⁺), F∘∇(uh⁺)')

createpvd(folder * "/" * pname) do pvd
  step = 0
  time = 0
  driverpost(pvd, step, time)
  println("Entering the time loop")
  while time < t_end

    step += 1
    time += Δt
    @printf "Step: %i\n" step
    @printf "Time: %.3f s\n" time
    
    #-----------------------------------------
    # Update boundary conditions
    #-----------------------------------------
    TrialFESpace!(Uφ, dirichlet_φ, time)
    TrialFESpace!(Uu, dirichlet_u, time)
    TrialFESpace!(Uθ, dirichlet_θ, time)

    println("Electric staggered step")
    op_elec = FEOperator(res_elec(time), jac_elec(time), Uφ, Vφ)
    solve!(φh⁺, solver, op_elec)
    
    println("Mechanical staggered step")
    op_mec = FEOperator(res_mec(time), jac_mec(time), Uu, Vu)
    solve!(uh⁺, solver, op_mec)
    
    println("Thermal staggered step")
    op_therm = FEOperator(res_therm(time), jac_therm(time), Uθ, Vθ)
    solve!(θh⁺, solver, op_therm)

    #-----------------------------------------
    # Post processing
    #-----------------------------------------
    driverpost(pvd, step, time)

    #-----------------------------------------
    # Update boundary conditions and old step
    #-----------------------------------------
    TrialFESpace!(Uφ⁻, dirichlet_φ, time)
    TrialFESpace!(Uu⁻, dirichlet_u, time)
    TrialFESpace!(Uθ⁻, dirichlet_θ, time)

    φ⁻ .= get_free_dof_values(φh⁺)
    u⁻ .= get_free_dof_values(uh⁺)
    θ⁻ .= get_free_dof_values(θh⁺)
    update_state!(update_η, η⁻, θh⁺, E∘∇(φh⁺),  F∘∇(uh⁺)')
  end
end


times = [0:Δt:t_end]
p1 = plot(times, ηtot, labels="Entropy", style=:solid, lcolor=:black, width=2)
p1 = plot!(twinx(p1), times, θavg, labels="Temperature", style=:dash, lcolor=:gray, width=2, xticks=false)
Ψint = Ψmec + Ψele + Ψthe
Ψtot = Ψint - Ψdir
p2 = plot(times, [Ψint Ψdir Ψtot], labels=["Ψu+Ψφ+Ψθ" "Ψφ,Dir" "Ψ"], style=[:solid :dash :solid], lcolor=[:black :black :gray], width=2)
p3 = plot(times, umax, labels="uz,L∞", color=:black, width=2)
p4 = plot(p1, p2, p3, layout=@layout([a b c]), size=(1200, 400))
display(p4)
