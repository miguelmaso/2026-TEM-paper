using HyperFEM
using HyperFEM.ComputationalModels.PostMetrics
using Gridap, Gridap.FESpaces, Gridap.Geometry
using GridapSolvers, GridapSolvers.NonlinearSolvers
using Printf
using JLD2
using MultiAssign


## Domain

function generate_tessellation(; width, thick0, prestretch, ndivisions, args...)
  λ3 = 1 / prestretch^2
  thick = thick0*λ3
  domain = (0.0, 0.5width, 0.0, 0.5width, 0.0, thick)
  partition = (ndivisions, ndivisions, ndivisions÷10)
  geometry = CartesianDiscreteModel(domain, partition)
  labels = get_face_labeling(geometry)
  add_tag_from_tags!(labels, "top",    CartesianTags.faceXY1)
  add_tag_from_tags!(labels, "bottom", CartesianTags.faceXY0)
  add_tag_from_tags!(labels, "faces", [CartesianTags.face1YZ⁺; CartesianTags.faceX1Z⁺])
  add_tag_from_tags!(labels, "x_sym", [CartesianTags.face0YZ; CartesianTags.edge0Y0; CartesianTags.edge0Y1])
  add_tag_from_tags!(labels, "y_sym", [CartesianTags.faceX0Z; CartesianTags.edgeX00; CartesianTags.edgeX01])
  add_tag_from_tags!(labels, "center_axis", CartesianTags.edge00Z⁺)
  add_tag_from_vertex_filter!(labels, "top_electrode", geometry,    p -> p[3] ≈ thick && abs(p[1]) <= 0.25width+1e-6 && abs(p[2]) <= 0.25width+1e-6)
  add_tag_from_vertex_filter!(labels, "bottom_electrode", geometry, p -> p[3] ≈ 0.0   && abs(p[1]) <= 0.25width+1e-6 && abs(p[2]) <= 0.25width+1e-6)
  geometry
end


## Constitutive model

function build_model(; θr, args...)
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

  # Yeoh model
  C10 = 1e4   # [Pa]
  C20 = -94.0 # [Pa]
  C30 = 0.82  # [Pa]

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

  coercive_volumetric = VolumetricEnergy(λ=κr)
  hyper_elastic_model = NonlinearMooneyRivlin3D(μ1=μe1, μ2=μe2, α1=α1, α2=α2, λ=0.0)
  hyper_elastic_model = Yeoh3D(C10=C10, C20=C20, C30=C30, λ=0.0)
  branch_1 = ViscousIncompressible(IsochoricNeoHookean3D(μ=μ1), τ=τ1)
  branch_2 = ViscousIncompressible(IsochoricNeoHookean3D(μ=μ2), τ=τ2)
  branch_3 = ViscousIncompressible(IsochoricNeoHookean3D(μ=μ3), τ=τ3)
  visco_model = GeneralizedMaxwell(hyper_elastic_model, branch_1, branch_2, branch_3)
  dielec_model = IdealDielectric(ε=εr*ε0)
  thermal_volumetric = ThermalVolumetric(coercive_volumetric, θr=θr, cv0=cv0, α=α, κ=κ, γ=γv)
  thermo_el = NonlinearMeltingLaw(θr=θr, θM=θ∞, γ=γ∞)
  thermo_vis = NonlinearSofteningLaw(θr=θr, θT=θα, γ=γα, δ=δα)
  thermo_dielec = NonlinearMeltingLaw(θr=θr, θM=θε, γ=γε)
  thermal_dielec = ThermoElectroModel(dielec_model, thermo_dielec)
  model = ThermoElectroMech_Bonet(thermal_volumetric, thermal_dielec, visco_model; el=thermo_el, vis=thermo_vis)
  return model
end


## Kinematics

struct PrestretchKinematics
  Fp
end

function PrestretchKinematics(; prestretch, args...)
  model = build_model(; args...)
  _, Pv, ∂Pv∂F = model.thermo.mechano()  # Volumetric penalty
  _, Pe, ∂Pe∂F = model.mechano.longterm()  # Deviatoric term
  get_Fp(λ3) = TensorValue{3,3}(prestretch, 0.0, 0.0, 0.0, prestretch, 0.0, 0.0, 0.0, λ3)
  P33(F) = Pv(F)[3,3] + Pe(F)[3,3]
  ∂P33(F) = ∂Pv∂F(F)[9,9] + ∂Pe∂F(F)[9,9]
  λ3 = 1/prestretch^2
  tol = 1e-10
  maxiter = 10
  for _ in 1:maxiter
    F_current = get_Fp(λ3)
    res = P33(F_current)
    if abs(res) < tol
      break
    end
    λ3 -= res / ∂P33(F_current)
  end
  return PrestretchKinematics(get_Fp(λ3))
end

function HyperFEM.get_Kinematics(::Type{Mechano}, k::PrestretchKinematics)
  F(∇u) = (I3 + ∇u)·k.Fp
  H(F) = cof(F)
  J(F) = det(F)
  return F, H, J
end

function HyperFEM.get_Kinematics(::Type{Electro}, k::PrestretchKinematics)
  E(∇φ) = -k.Fp'·∇φ
end

function evaluatebc(bc, i, t, x=Point(0.0, 0.0, 0.0))
  bc.values[i](t)(x)
end


## FEM solver

function solve_problem(data)
  
  pname = stem(@__FILE__)
  folder = abspath(dirname(@__FILE__), "results_$(data.ndivisions)")
  outpath = joinpath(folder, pname)
  setupfolder(folder; remove=".vtu")

  model = build_model(; data...)

  k = PrestretchKinematics(; data...)
  F, H, J = get_Kinematics(Mechano, k)
  E       = get_Kinematics(Electro, k)
  ∇u0   = TensorValue(ntuple(_ -> 0.0, 9))
  Fp    = F(∇u0)
  ∂F∂∇u = Fp
  ∂E∂∇φ = -Fp'
  invJp = 1/J(Fp)
  
  geometry = generate_tessellation(; data...)

  # Discrete domain, integration and boundary conditions
  Δt = data.Δt
  t_end = data.t_end
  order = data.order
  degree = 2 * order
  Ω = Triangulation(geometry)
  dΩ = Measure(Ω, degree)

  solver_mech  = FESolver(NewtonSolver(LUSolver(); maxiter=20, atol=1e-8,  rtol=1e-8,  verbose=true))
  solver_elec  = FESolver(NewtonSolver(LUSolver(); maxiter=20, atol=1e-10, rtol=1e-10, verbose=true))
  solver_therm = FESolver(NewtonSolver(LUSolver(); maxiter=20, atol=1e-10, rtol=1e-10, verbose=true))

  reffeu = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
  reffeφ = ReferenceFE(lagrangian, Float64, order)
  reffeθ = ReferenceFE(lagrangian, Float64, order)

  Vu = TestFESpace(geometry, reffeu, data.dirichlet_u, conformity=:H1, dirichlet_masks=data.dirichlet_u_masks)
  Vφ = TestFESpace(geometry, reffeφ, data.dirichlet_φ, conformity=:H1)
  Vθ = TestFESpace(geometry, reffeθ, data.dirichlet_θ, conformity=:H1)

  println("======================================")
  println("Number of divisions :           $(data.ndivisions)")
  println("Mechanical degrees of freedom : $(Vu.nfree)")
  println("Electrical degrees of freedom : $(Vφ.nfree)")
  println("Thermal degrees of freedom :    $(Vθ.nfree)")
  println("Total degrees of freedom :      $(Vu.nfree+Vφ.nfree+Vθ.nfree)")
  println("======================================")

  # Trial FE spaces and state variables

  Uu  = TrialFESpace(Vu, data.dirichlet_u)
  Uφ  = TrialFESpace(Vφ, data.dirichlet_φ)
  Uθ  = TrialFESpace(Vθ, data.dirichlet_θ)
  uh⁺ = FEFunction(Uu, zero_free_values(Uu))
  φh⁺ = FEFunction(Uφ, zero_free_values(Uφ))
  θh⁺ = FEFunction(Uθ, data.θr * ones(Vθ.nfree))

  Uu⁻ = TrialFESpace(Vu, data.dirichlet_u)
  Uφ⁻ = TrialFESpace(Vφ, data.dirichlet_φ)
  Uθ⁻ = TrialFESpace(Vθ, data.dirichlet_θ)
  uh⁻ = FEFunction(Uu⁻, zero_free_values(Uu))
  φh⁻ = FEFunction(Uφ⁻, zero_free_values(Uφ))
  θh⁻ = FEFunction(Uθ⁻, data.θr * ones(Vθ.nfree))

  η⁻  = CellState(0.0, dΩ)
  D⁻  = CellState(0.0, dΩ)

  Eh  = E∘∇(φh⁺)
  Eh⁻ = E∘∇(φh⁻)
  Fh  = F∘∇(uh⁺)'
  Fh⁻ = F∘∇(uh⁻)'
  A   = CellState(model, Fp / J(Fp), dΩ)

  # Weak forms: residual and jacobian

  Ψ, ∂Ψ∂F, ∂Ψ∂E, ∂Ψ∂θ, ∂∂Ψ∂FF, ∂∂Ψ∂EE, ∂∂Ψ∂θθ, ∂∂Ψ∂FE, ∂∂Ψ∂Fθ, ∂∂Ψ∂Eθ = model()
  D, ∂D∂θ = Dissipation(model)
  η(x...) = -∂Ψ∂θ(x...)
  ∂η∂θ(x...) = -∂∂Ψ∂θθ(x...)
  update_η(_, F, E, θ, Fn, A...) = (true, η(F, E, θ, Fn, A...))
  update_D(_, F, E, θ, Fn, A...) = (true, D(F, E, θ, Fn, A...))
  κ = model.thermo.thermo.κ


  # Electro
  res_elec(Λ) = (φ, vφ) -> -1.0*∫(invJp * (∇(vφ)·Fp) ⋅ (∂Ψ∂E ∘ (F∘(∇(uh⁺)'), E∘(∇(φ)), θh⁺, Fh⁻, A...)))dΩ
  jac_elec(Λ) = (φ, dφ, vφ) -> -1.0*∫(invJp * (∇(vφ)·Fp) ⋅ ((∂∂Ψ∂EE ∘ (F∘(∇(uh⁺)'), E∘(∇(φ)), θh⁺, Fh⁻, A...)) ⋅ (∂E∂∇φ·∇(dφ))))dΩ

  # Mechano
  res_mec(Λ) = (u, v) -> ∫(invJp * (∇(v)'·Fp) ⊙ (∂Ψ∂F ∘ (F∘(∇(u)'), E∘(∇(φh⁺)), θh⁺, Fh⁻, A...)))dΩ
  jac_mec(Λ) = (u, du, v) -> ∫(invJp * (∇(v)'·Fp) ⊙ ((∂∂Ψ∂FF ∘ (F∘(∇(u)'), E∘(∇(φh⁺)), θh⁺, Fh⁻, A...)) ⊙ (∇(du)'·∂F∂∇u)))dΩ

  # Thermo
  res_therm(Λ) = (θ, vθ) -> begin (
    1/Δt*∫( invJp * (θ*(η∘(F∘∇(uh⁺)', E∘∇(φh⁺), θ, Fh⁻, A...)) -θh⁻*η⁻)*vθ )dΩ +
    -1/Δt*0.5*∫( invJp * (η∘(F∘∇(uh⁺)', E∘∇(φh⁺), θ, Fh⁻, A...) + η⁻)*(θ - θh⁻)*vθ )dΩ +
    -0.5*∫( invJp * (D∘(F∘∇(uh⁺)', E∘∇(φh⁺), θ, Fh⁻, A...) + D⁻)*vθ )dΩ +
    0.5*∫( invJp * κ*∇(θ)·∇(vθ) + κ*∇(θh⁻)·∇(vθ) )dΩ
  )
  end
  jac_therm(Λ) = (θ, dθ, vθ) -> begin (
    1/Δt*∫( invJp * (η∘(F∘∇(uh⁺)', E∘∇(φh⁺), θ, Fh⁻, A...) + θ*(∂η∂θ∘(F∘∇(uh⁺)', E∘∇(φh⁺), θ, Fh⁻, A...)))*dθ*vθ )dΩ +
    -1/Δt*0.5*∫( invJp * (∂η∂θ∘(F∘∇(uh⁺)', E∘∇(φh⁺), θ, Fh⁻, A...)*(θ - θh⁻) + η∘(F∘∇(uh⁺)', E∘∇(φh⁺), θ, Fh⁻, A...) + η⁻)*dθ*vθ )dΩ +
    -0.5*∫( invJp * (∂D∂θ∘(F∘∇(uh⁺)', E∘∇(φh⁺), θ, Fh⁻, A...))*dθ*vθ )dΩ +
    ∫( invJp * 0.5*κ*∇(dθ)·∇(vθ) )dΩ
  )
  end

  # Post-processor

  fields = (:time, :Ψmec, :Ψele, :Ψthe, :Ψdir, :Dvis, :ηtot, :θavg, :λ, :V, :∂Pθ_F, :∂Dθ_E, :cv)
  metrics = NamedTuple{fields}(Float64[] for _ in 1:length(fields))

  function post_metrics!(data, step, time)
    b_φ = assemble_vector(vφ -> res_elec(time)(φh⁺, vφ), DirichletFESpace(Vφ))[:]
    ∂φt_fix = (get_dirichlet_dof_values(Uφ) - get_dirichlet_dof_values(Uφ⁻)) / Δt
    θ1h = FEFunction(Vθ, ones(Vθ.nfree))
    push!(data.time, time)
    push!(data.Ψmec, sum(res_mec(time)(uh⁺, uh⁺-uh⁻))/Δt)
    push!(data.Ψele, sum(res_elec(time)(φh⁺, φh⁺-φh⁻))/Δt)
    push!(data.Ψthe, sum(res_therm(time)(θh⁺, θ1h)))
    push!(data.Ψdir, b_φ · ∂φt_fix)
    push!(data.Dvis, sum(∫( D∘(Fh, Eh, θh⁺, Fh⁻, A...) )dΩ))
    push!(data.ηtot, sum(∫( η∘(Fh, Eh, θh⁺, Fh⁻, A...) )dΩ))
    push!(data.θavg, sum(∫( θh⁺ )dΩ) / sum(∫(1)dΩ))
    umax = component_LInf(uh⁺, :x, Ω)
    push!(data.λ, (1+umax/problem_data.width*4)*problem_data.prestretch)
    push!(data.V, evaluatebc(problem_data.dirichlet_φ, 1, time))
    push!(data.∂Pθ_F, sum(∫( (∂∂Ψ∂Fθ∘(Fh, Eh, θh⁺, Fh⁻, A...))⊙(Fh-Fh⁻)/Δt )dΩ))
    push!(data.∂Dθ_E, sum(∫( -(∂∂Ψ∂Eθ∘(Fh, Eh, θh⁺, Fh⁻, A...))⋅(Eh-Eh⁻)/Δt )dΩ))
    push!(data.cv,    sum(∫( -(∂∂Ψ∂θθ∘(Fh, Eh, θh⁺, Fh⁻, A...)) )dΩ))
  end

  function post_vtk!(pvd, step, time)
    if mod(step, 5) == 0
      ηh = interpolate_L2_scalar(η∘(Fh, Eh, θh⁺, Fh⁻, A...), Ω, dΩ)
      Jh = interpolate_L2_scalar(J∘Fh, Ω, dΩ)
      pvd[time] = createvtk(Ω, outpath * @sprintf("_%03d", step), cellfields=["u" => uh⁺, "ϕ" => φh⁺, "θ" => θh⁺, "η" => ηh, "J" => Jh])
    end
  end

  # Time integration

  update_time_step!(model, Δt)
  update_state!(update_η, η⁻, Fh, Eh, θh⁺, Fh⁻, A...)
  update_state!(update_D, D⁻, Fh, Eh, θh⁺, Fh⁻, A...)

  createpvd(outpath) do pvd
    u⁻ = get_free_dof_values(uh⁻)
    φ⁻ = get_free_dof_values(φh⁻)
    θ⁻ = get_free_dof_values(θh⁻)
    step = 0
    time = 0.0
    post_vtk!(pvd, step, time)
    post_metrics!(metrics, step, time)
    println("Entering the time loop")
    try
      while time < t_end
        step += 1
        time += Δt
        printstyled(@sprintf("Step: %i\nTime: %.3f s\n", step, time), color=:green, bold=true)

        #-----------------------------------------
        # Update boundary conditions
        #-----------------------------------------
        TrialFESpace!(Uφ, data.dirichlet_φ, time)
        TrialFESpace!(Uu, data.dirichlet_u, time)
        TrialFESpace!(Uθ, data.dirichlet_θ, time)

        println("Electric staggered step")
        op_elec = FEOperator(res_elec(time), jac_elec(time), Uφ, Vφ)
        Gridap.solve!(φh⁺, solver_elec, op_elec)

        println("Mechanical staggered step")
        op_mech = FEOperator(res_mec(time), jac_mec(time), Uu, Vu)
        Gridap.solve!(uh⁺, solver_mech, op_mech)

        println("Thermal staggered step")
        op_therm = FEOperator(res_therm(time), jac_therm(time), Uθ, Vθ)
        Gridap.solve!(θh⁺, solver_therm, op_therm)

        #-----------------------------------------
        # Post processing
        #-----------------------------------------
        post_vtk!(pvd, step, time)
        post_metrics!(metrics, step, time)

        #-----------------------------------------
        # Update boundary conditions and old step
        #-----------------------------------------
        update_state!(update_η, η⁻, Fh, Eh, θh⁺, Fh⁻, A...)
        update_state!(update_D, D⁻, Fh, Eh, θh⁺, Fh⁻, A...)
        update_state!(model, A, Fh, Eh, θh⁺, Fh⁻)

        TrialFESpace!(Uφ⁻, data.dirichlet_φ, time)
        TrialFESpace!(Uu⁻, data.dirichlet_u, time)
        TrialFESpace!(Uθ⁻, data.dirichlet_θ, time)

        φ⁻ .= get_free_dof_values(φh⁺)
        u⁻ .= get_free_dof_values(uh⁺)
        θ⁻ .= get_free_dof_values(θh⁺)

        GC.gc()
      end
    catch e
      rethrow(e)
    finally
      @save "$(outpath)_metrics_$(data.prestretch).jld2" metrics  # Save the time evolution
      @save "$(outpath)_uh_$(data.order)_$(data.ndivisions).jld2" uh⁺  # Save the final state
    end
  end
  return (; metrics, uh⁺)
end
