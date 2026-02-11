#------------------------------------------
#------------------------------------------
#
# Calibration of VHB 4905 polymer.
# Data and libraries from:
# https://doi.org/10.1016/j.ijnonlinmec.2019.103263 - Liao, Mokarram et al., 2022, On thermo-viscoelastic experimental characterization and numerical modelling of VHB polymer
# https://doi.org/10.1002/zamm.201400110 - Dippel et al., 2015, Thermo-mechanical couplings in elastomers - experiments and modelling
# https://docs.sciml.ai/Optimization/stable/#Citation - Kumar, 2023, Optimization.jl: A unified optimization package
#
#------------------------------------------
#------------------------------------------

using Plots, Printf
using HyperFEM, HyperFEM.ComputationalModels.EvolutionFunctions
using Optimization, OptimizationOptimJL, OptimizationMetaheuristics, Optim
using LinearAlgebra, FiniteDiff, Distributions

include("ConstitutiveModelling.jl")
include("ExperimentsData.jl")

const colors2 = mapreduce(c -> [c,c], vcat, palette(:default))
const colors3 = mapreduce(c -> [c,c,c], vcat, palette(:default))
const colors4 = mapreduce(c -> [c,c,c,c], vcat, palette(:default))

const temp_label = data -> @sprintf("%2.0fºC", data.θ-K0)
const vel_label = data -> @sprintf("%.2f/s", data.v)
const stretch_label = data -> @sprintf("%.0f%%", 100*(data.λ_max-1))

const label_λσ = (; xlabel="Stretch [-]", ylabel="Stress [Pa]")
default(titlefontsize=10)

##-----------------------------------------
# Objective function
#------------------------------------------

function loss(model::PhysicalModel, data::LoadingTest)
  σ_model = simulate_experiment(model, data.θ, data.Δt, data.λ)
  σ_err = (σ_model .- data.σ) / data.σ_max
  sum(abs2, σ_err) / length(σ_err) * data.weight
end

function loss(model::PhysicalModel, data::HeatingTest)
  cv_model = simulate_experiment(model, data.θ)
  cv_err = (cv_model .- data.cv) / data.cv_max
  sum(abs2, cv_err) / length(cv_err) * data.weight
end

function loss(model::PhysicalModel, data::Vector{<:ExperimentData})
  sum(d -> loss(model, d), data) / sum(d -> d.weight, data)
end

function loss(params, data)
  model = build_constitutive_model(params...)
  loss(model, data)
end

function experiment_prediction(model::PhysicalModel, data::LoadingTest)
  y_true = data.σ
  y_pred = simulate_experiment(model, data.θ, data.Δt, data.λ)
  return y_true, y_pred
end

function experiment_prediction(model::PhysicalModel, data::HeatingTest)
  y_true = data.cv
  y_pred = simulate_experiment(model, data.θ)
  return y_true, y_pred
end

function experiment_prediction(model::PhysicalModel, data::Vector{<:ExperimentData})
  y = map(d -> experiment_prediction(model, d), data)
  return vcat(first.(y)...), vcat(last.(y)...)
end


function r2(params, data)
  model = build_constitutive_model(params...)
  y_true, y_pred = experiment_prediction(model, data)
  y_mean = mean(y_true)
  ss_res = sum(abs2, y_true .- y_pred)
  ss_tot = sum(abs2, y_true .- y_mean)
  return 1 - (ss_res / ss_tot)
end

function covariance_matrix(model_builder, params, data)
  n_params = length(params)
  n_data = npoints(data)
  n_dof = n_data - n_params

  sse_val = loss(params, data)
  res_variance = sse_val / n_dof
  
  # Compute the covariance matrix
  H = FiniteDiff.finite_difference_hessian(p -> loss(p, data), params)
  local cov_matrix
  try
    cov_matrix = 2*res_variance*inv(H)
  catch
    println("⚠️ Singular hessian matrix. Probably there are redundant parameters.")
    return
  end
  cov_matrix, H
end

function stats(model_builder, params, data, names=map("",params))
  n_dof = npoints(data) - length(params)
  sse_val = loss(params, data)
  cov_matrix, H = covariance_matrix(model_builder, params, data)

  t_crit = quantile(TDist(n_dof), 0.975) # t-Student value
  std_errs = sqrt.(abs.(diag(cov_matrix)))
  ci_lower = params .- t_crit .* std_errs
  ci_upper = params .+ t_crit .* std_errs

  r(num) = round(num, sigdigits=2)
  for i in eachindex(params)
    println("$(names[i]) : $(r(params[i])) ± $(r(t_crit*std_errs[i])) ()")
    println("     Interval : [$(r(ci_lower[i])) , $(r(ci_upper[i]))]")
    sens = H[i,i] * params[i]^2 / sse_val
    println("     Sensitivity : $(r(sens))")
  end
  println("R2 : ", lpad(@sprintf("%.1f", 100*r2(params,data)), 8))
  return ci_lower, ci_upper
end

function covariance_uncertainity(model_builder, params, data, n_samples=100)
  cov_matrix, _ = covariance_matrix(model_builder, params, data)
  M = (cov_matrix + cov_matrix') / 2
  vals, vecs = eigen(M)
  
  threshold = maximum(abs.(vals)) * 1e-8  # Security threshold (adjustable)
  vals_clean = max.(vals, threshold) # We must enforce positivity of the eigenvalues
  if any(vals .<= 0)
    println("⚠️  Repairing covariance matrix:")
    println("   Original eigenvalues: ", round.(vals, sigdigits=3))
    println("   New eigenvalues:      ", round.(vals_clean, sigdigits=3))
  end
  
  M_reconstructed = vecs * Diagonal(vals_clean) * vecs'
  S_final = Symmetric(M_reconstructed)
  mv_param_dist = MvNormal(params, S_final)
  rand(mv_param_dist, n_samples) # Population of n samples of parameters sets
end

function bootstrap_uncertainity(model_builder, params, data, n_samples=20)
  # Estimating standard error (assuming gaussian noise)
  base_model = model_builder(params...)
  y_true, y_pred = experiment_prediction(base_model, data)
  residuals = y_pred .- y_true
  sigma_err = std(residuals)
  println("Estimated noise on data: ", sigma_err)

  bootstrapped_params = Vector{Vector{Float64}}()
  
  println("Starting bootstrapping ($n_samples samples)")
  Threads.@threads for _ in 1:n_samples
    # A. Generating synthetic data (theroetical curve + noise)
    synthetic_data = map(d -> begin
      T = typeof(d)
      y_true, y_clean = experiment_prediction(base_model, d)
      y_noise = y_clean .+ randn(length(y_clean)) * sigma_err
      return T(d, y_noise)
    end, data)

    opt_func = OptimizationFunction(loss)
    opt_prob = OptimizationProblem(opt_func, params, synthetic_data)#, lb=lb, ub=ub)
    sol = solve(opt_prob, NelderMead(), maxiters=100) # Low maxiter because we start close
    
    push!(bootstrapped_params, sol.u)
    print(".") # progress bar
  end
  println("\nBootstrapping completed")
  return hcat(bootstrapped_params...)
end


function plot_confidence_bands!(model_builder, params, random_params, data)
  for p_sample in eachcol(random_params)
    model = model_builder(p_sample...)
    σ_sim = simulate_experiment(model, data.θ, data.Δt, data.λ)
    plot!(p, data.λ, σ_sim, color=:blue, alpha=0.05, lw=1, label="")
  end

  model_opt = model_builder(params...)
  σ_opt = simulate_experiment(model_opt, data.θ, data.Δt, data.λ)
  plot!(p, data.λ, σ_opt, color=:blue, lw=2, label="Model")
  
  scatter!(p, data.λ, data.σ, label="Experiment", color=:black, markerstrokewidth=0)
end


build_constitutive_model(cv0, γv) = 
  yeoh_1_branch(1.0e4, 1.0e2, 1.0e0, 4.0e4, 1.0, cv0, γv, 0.5, 0.5, 0.0, 0.0)

build_constitutive_model(C1, C2, C3, μ1, p1) = 
  yeoh_1_branch(C1, C2, C3, μ1, p1, 1283.88, 0.78, 0.0, 0.0, 0.0, 0.0)

build_constitutive_model(C1, C2, C3, μ1, p1, γel, γvis, δel=0.0, δvis=0.0) = 
  yeoh_1_branch(C1, C2, C3, μ1, p1, 1283.88, 0.78, γel, γvis, δel, δvis)

function yeoh_1_branch(C1, C2, C3, μ1, p1, cv0, γv, γel, γvis, δel, δvis)
  long_term = Yeoh3D(λ=0.0, C10=C1, C20=C2, C30=C3)
  branch_1 = ViscousIncompressible(IncompressibleNeoHookean3D(λ=0., μ=μ1), τ=exp(p1))
  visco_elasto = GeneralizedMaxwell(long_term, branch_1)
  thermal_model = ThermalModel(Cv=cv0, θr=θr, α=αr, κ=1.0)
  func_v = VolumetricLaw(θr, γv)
  func_el = InterceptLaw(θr, γel, δel)
  func_vis = InterceptLaw(θr, γvis, δvis)
  return ThermoMech_Bonet(thermal_model, visco_elasto, func_v, func_el, func_vis)
end

function neo_hookean_1_branch(μe, μ1, p1, cv0, γv, γel, γvis, δel, δvis)
  long_term = NeoHookean3D(λ=0.0, μ=μe)
  branch_1 = ViscousIncompressible(IncompressibleNeoHookean3D(λ=0., μ=μ1), τ=exp(p1))
  visco_elasto = GeneralizedMaxwell(long_term, branch_1)
  thermal_model = ThermalModel(Cv=cv0, θr=θr, α=αr, κ=1.0)
  func_v = VolumetricLaw(θr, γv)
  func_el = InterceptLaw(θr, γel, δel)
  func_vis = InterceptLaw(θr, γvis, δvis)
  return ThermoMech_Bonet(thermal_model, visco_elasto, func_v, func_el, func_vis)
end


##-----------------------------------------
# Experiments data
#------------------------------------------

heating_data = read_data(joinpath(@__DIR__, "Dippel 2015.csv"), HeatingTest)
mechanical_data = read_data(joinpath(@__DIR__, "Liao_Mokarram 2022.csv"), LoadingTest)
foreach(r -> r.θ < K0-10 && (r.weight *= 0.5), mechanical_data)
foreach(r -> r.θ > K0+70 && (r.weight *= 0.5), mechanical_data)
foreach(r -> r.λ_max < 4 && (r.weight *= 2.0), mechanical_data)
filter!(r -> (r.θ > -10+K0), mechanical_data)
println(heating_data)
println(mechanical_data)


##-----------------------------------------
# Thermal characterization
#------------------------------------------
function thermal_characterization(data)
  #    [  cv0,  γv]
  p0 = [1.0e3, 0.5]  # Initial seed
  lb = [ 10.0, 0.0]  # Minimum search limits
  ub = [1.0e5, 1.0]  # Maximum search limits
  opt_func = OptimizationFunction(loss)   # AutoFiniteDiff() is needed to be provided for gradient-based search algorithms
  opt_prob = OptimizationProblem(opt_func, p0, data, lb=lb, ub=ub)
  solve(opt_prob, Optim.ParticleSwarm(lower=lb, upper=ub, n_particles=100), maxiters=1000, maxtime=60.0)  # ECA (Evolutionary Centers Algorithm), NelderMead, LBFGS
end

function plot_experiment!(model, data::HeatingTest)
  cv_values = simulate_experiment(model, data.θ)
  plot!(data.θ.-K0, [cv_values, data.cv], label=["Model" "Experiment"], xlabel="T [ºC]", ylabel="cv [J/(kg·ºK)]", lw=2, mark=[:none :circle], markerstrokewidth=0)
end

sol_heat = thermal_characterization(heating_data)
stats(build_constitutive_model, sol_heat, heating_data, ["cv0", "γv"])

cv0, γv = sol_heat.u
model = build_constitutive_model(cv0, γv)
text1 = text("cv⁰ = " * @sprintf("%.0f", cv0) * " N/(m²·K)\n" *
             "γ̄   = " * @sprintf("%.2f", γv) * "\n" *
             "R²  = " * @sprintf("%.0f", 100*r2(sol_heat.u, heating_data)) * " %",
             8, :left)

# Plot the solution
p = plot(title="Volumetric characterization"; label_λσ...)
plot_experiment!(model, heating_data[1])
annotate!((0.05, 0.75), text1, relative=true)
display(p);


##-----------------------------------------
# Reference characterization
#------------------------------------------
function viscoelastic_characterization(data)
  #    [   C1,     C2,     C3,      μ1     p1]
  p0 = [  3e4,   -2e2,    3e0,   5.0e4,   0.0]  # Initial seed
  lb = [1.0e4, -2.0e3,  1.0e0,   1.0e4,  -5.0]  # Minimum search limits
  ub = [2.0e5,  2.0e3,  2.0e2,   1.0e5,   5.0]  # Maximum search limits
  opt_func = OptimizationFunction(loss)
  opt_prob = OptimizationProblem(opt_func, p0, data, lb=lb, ub=ub)
  sol = solve(opt_prob, ParticleSwarm(lower=lb, upper=ub, n_particles=100), maxiters=1000, maxtime=60.0)

  opt_prob = OptimizationProblem(opt_func, sol.u, data)
  sol = solve(opt_prob, NelderMead())
end

subset_T20 = filter(r -> r.θ ≈ 20+K0, mechanical_data)
sol_mech = viscoelastic_characterization(subset_T20)
l_mech, u_mech = stats(build_constitutive_model, sol_mech.u, subset_T20, ["C10", "C20", "C30", "μ1", "p1"])
model  = build_constitutive_model(sol_mech.u...)
text2 = text(@sprintf("R² = %.1f %%", 100*r2(sol_mech.u, subset_T20)), 8, :left)

function plot_reference_config!(model, data, labelfn=_->"")
  σ_values = simulate_experiment(model, data.θ, data.Δt, data.λ)
  label = labelfn(data)
  plot!(data.λ, [σ_values, data.σ], label=[label ""], typ=[:path :scatter], lw=2, mswidth=0, color_palette=colors2)
end

p = plot(title="20ºC, 0.1/s, 300%\n95% confidence bands"; label_λσ...)
experim = getfirst(r -> r.v≈0.1 && r.λ_max≈4.0, subset_T20)
# rand_params = covariance_uncertainity(build_constitutive_model, sol_mech.u, subset_T20)
rand_params = bootstrap_uncertainity(build_constitutive_model, sol_mech.u, subset_T20)
plot_confidence_bands!(build_constitutive_model, sol_mech.u, rand_params, experim)
annotate!((0.05, 0.75), text2, relative=true)
display(p);

p = plot(title="20ºC, 0.1/s"; label_λσ...)
for e in filter(r -> r.v ≈ 0.1, subset_T20)
  plot_reference_config!(model, e, stretch_label)
end
annotate!((0.05, 0.75), text2, relative=true)
display(p);

p = plot(title="20ºC, 300%"; label_λσ...)
for e in filter(r -> r.λ_max ≈ 4.0, subset_T20)
  plot_reference_config!(model, e, vel_label)
end
annotate!((0.05, 0.75), text2, relative=true)
display(p);


##-----------------------------------------
# Visco-elastic characterization
#------------------------------------------
function mechanical_characterization(data)
  #    [   C1,     C2,     C3,      μ1     p1,    γel,   γvis] #,   δel,  δvis]
  p0 = [  3e4,   -2e2,    3e0,   5.0e4,   0.0,    1.0,    1.0] #,   0.1,   0.1]  # Initial seed
  lb = [2.0e4, -2.0e3, -2.0e1,   1.0e4,  -5.0,  -10.0,  -10.0] #,  -1.0,  -1.0]  # Minimum search limits
  ub = [2.0e5,  2.0e3,  2.0e1,   1.0e5,   5.0,  100.0,  100.0] #,   1.0,   1.0]  # Maximum search limits
  opt_func = OptimizationFunction(loss)   # AutoFiniteDiff() is needed for gradient-based search algorithms
  opt_prob = OptimizationProblem(opt_func, p0, data, lb=lb, ub=ub)
  solve(opt_prob, ParticleSwarm(lower=lb, upper=ub, n_particles=100), maxiters=1000, maxtime=60.0)  # ECA (Evolutionary Centers Algorithm), NelderMead, LBFGS
end

function plot_experiment!(model, data::LoadingTest, labelfn=d->"")
  σ_values = simulate_experiment(model, data.θ, data.Δt, data.λ)
  label = labelfn(data)
  plot!(data.λ, [σ_values, data.σ], label=[label ""], typ=[:path :scatter], lw=2, mswidth=0, color_palette=colors2)
end

sol_mech = mechanical_characterization(mechanical_data)
stats(build_constitutive_model, sol_mech.u, mechanical_data, ["C10", "C20", "C30", "μ1", "p1", "γel", "γvis"])#, "δel", "δvis"])

C1, C2, C3, μ1, p1, γel, γvis = sol_mech.u
model = build_constitutive_model(sol_mech.u...)
text2 = text(" γ̂el = " * @sprintf("%.1f\n", γel) *
             "γ̂vis = " * @sprintf("%.1f\n", γvis) *
             "  R² = " * @sprintf("%.1f %%", 100*r2(sol_mech.u, mechanical_data)),
             8, :left)

subset = filter(r -> (r.v ≈ 0.1 && r.λ_max ≈ 2 && r.θ > -10+K0), mechanical_data)
sort!(subset, by = r -> r.θ)
p = plot(title="0.1/s, 100%")
for e ∈ subset
  plot_experiment!(model, e, temp_label)
end
plot!([], [], label="Experiment", color=:black, typ=:scatter, wswidth=0)
plot!([], [], label="Model",      color=:black, lw=2)
annotate!((0.05, 0.5), text2, relative=true)
display(p);

subset = filter(r -> (r.v ≈ 0.1 && r.λ_max ≈ 4 && r.θ > -10+K0), mechanical_data)
sort!(subset, by = r -> r.θ)
p = plot(title="0.1/s, 300%")
for e ∈ subset
  plot_experiment!(model, e, temp_label)
end
plot!([], [], label="Experiment", color=:black, typ=:scatter, wswidth=0)
plot!([], [], label="Model",      color=:black, lw=2)
annotate!((0.05, 0.5), text2, relative=true)
display(p);

p = plot()
cons_model = yeoh_1_branch(20.e3, -630., 20., 54.e3, log(12.2), 1280.0, 0.78, 1.0, 3.0, 0.0, 0.0)
# cons_model = neo_hookean_1_branch(1.3e4, 4.4e4, 2.2, 1280.0, 0.78, 0.0, 0.0, 0.0, 0.0)
λ_max = 4.0
t_max = (λ_max-1) / 0.1
Δt = t_max / 20
λ_values = map(1 + (λ_max-1)*triangular(t_max), 0:Δt:1.6*t_max)
for T in [0.0, 20.0, 40.0]
  σ_values = simulate_experiment(cons_model, T+K0, Δt, λ_values) / 1e3
  plot!(λ_values, σ_values, label="T = $T ºC", xlabel="Stretch [-]", ylabel="Stress [kPa]", lw=2)
end
display(p);
