using HyperFEM
using HyperFEM.ComputationalModels.PostMetrics
using HyperFEM.ComputationalModels.CartesianTags
using HyperFEM.ComputationalModels.EvolutionFunctions
using Gridap, Gridap.FESpaces
using GridapSolvers, GridapSolvers.NonlinearSolvers
using GridapDistributed, GridapPETSc
using Printf
using Plots
using MultiAssign
using MPI
import Plots:mm

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

if rank == 0
    println("============================================")
    println("Hola, soy el proceso maestro (Rank 0).")
    println("El tamaño total del mundo MPI es: $size")
    # println("La partición requerida es: $(prod(rank_partition))")
    println("============================================")
end

pname = stem(@__FILE__)
folder = joinpath(@__DIR__, "results")
outpath = joinpath(folder, pname)
setupfolder(folder; remove=".vtu")

function simulation_run(rank_partition, distribute)
  parts = distribute(LinearIndices((prod(rank_partition),)))
  size = 0.1  # m
  ndivisions = 3
  domain = (0.0, size, 0.0, size, 0.0, size)
  partition = (ndivisions, ndivisions, ndivisions)
  geometry = CartesianDiscreteModel(parts, rank_partition, domain, partition)
  labels = get_face_labeling(geometry)
  add_tag_from_tags!(labels, "bottom", CartesianTags.faceZ0)
  add_tag_from_tags!(labels, "top", CartesianTags.faceZ1)
  add_tag_from_tags!(labels, "edge", CartesianTags.edgeX00)
  add_tag_from_tags!(labels, "corner", CartesianTags.corner000)

  # Constitutive model parameters
  ε  = 1.0
  μ1 = 1.0
  μ2 = 1.0
  λ  = 10.0
  τ1 = 0.8
  Cv = 0.1 # 17.385
  θr = 293.15
  κ  = λ + 2(μ1+μ2)
  α  = 22.33e-5 * κ
  γv = 1.0
  γd = 1.0

  # Constitutive model
  hyper_elastic_model = NeoHookean3D(λ=λ, μ=0.1μ1)
  branch1 = ViscousIncompressible(IncompressibleNeoHookean3D(λ=0.0, μ=μ1), τ=τ1)
  visco_model = GeneralizedMaxwell(hyper_elastic_model, branch1)
  elec_model = IdealDielectric(ε=ε)
  therm_model = ThermalModel(Cv=Cv, θr=θr, α=α, κ=κ, γv=γv, γd=γd)
  cons_model = ThermoElectroMech_Bonet(therm_model, elec_model, visco_model)
  ku = Kinematics(Mechano, Solid)
  ke = Kinematics(Electro, Solid)
  kt = Kinematics(Thermo, Solid)
  F, H, J = get_Kinematics(ku)
  E       = get_Kinematics(ke)

  # Setup integration
  order = 2
  degree = 2 * order
  Ω = Triangulation(geometry)
  dΩ = Measure(Ω, degree)
  t_end = 2.0  # s
  Δt = 0.02    # s
  update_time_step!(cons_model, Δt)

  # Dirichlet boundary conditions 
  dir_u_tags = ["corner", "edge", "bottom"]  # The first tag will overwrite the last one.
  dir_u_values = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
  dir_u_timesteps = [Λ->1, Λ->1, Λ->1]
  dir_u_masks = [[true,true,true], [false,true,true], [false,false,true]]
  dirichlet_u = DirichletBC(dir_u_tags, dir_u_values, dir_u_timesteps)

  voltage = 0.065
  dir_φ_tags = ["bottom", "top"]
  dir_φ_values = [0.0, voltage]
  dir_φ_timesteps = [Λ->1, triangular(1.0)]
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

  println("======================================")
  println("Mechanical degrees of freedom : $(Vu.nfree)")
  println("Electrical degrees of freedom : $(Vφ.nfree)")
  println("Thermal degrees of freedom :    $(Vθ.nfree)")
  println("Total degrees of freedom :      $(Vu.nfree+Vφ.nfree+Vθ.nfree)")
  println("======================================")

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

  η⁻  = CellState(0.0, dΩ)
  D⁻  = CellState(0.0, dΩ)

  Eh  = E∘∇(φh⁺)
  Eh⁻ = E∘∇(φh⁻)
  Fh  = F∘∇(uh⁺)'
  Fh⁻ = F∘∇(uh⁻)'
  A   = initialize_state(visco_model, dΩ)

  # =================================
  # Weak forms: residual and jacobian
  # =================================

  Ψ, ∂Ψ∂F, ∂Ψ∂E, ∂Ψ∂θ, ∂∂Ψ∂FF, ∂∂Ψ∂EE, ∂∂Ψ∂θθ, ∂∂Ψ∂FE, ∂∂Ψ∂Fθ, ∂∂Ψ∂Eθ = cons_model()
  D, ∂D∂θ = Dissipation(cons_model)
  η(x...) = -∂Ψ∂θ(x...)
  ∂η∂θ(x...) = -∂∂Ψ∂θθ(x...)
  update_η(_, θ, E, F, Fn, A) = (true, η(F, E, θ, Fn, A))
  update_D(_, θ, E, F, Fn, A) = (true, D(F, E, θ, Fn, A))
  κ = cons_model.thermo.κ

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

  ls = LUSolver()
  nls = NewtonSolver(ls; maxiter=20, atol=1e-10, rtol=1e-10, verbose=true)
  solver = FESolver(nls)

  # Postprocessor to save results
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
end

rank_partition = (2,2,2)
with_mpi() do distribute
  simulation_run(rank_partition,distribute)
end
