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

#------------------------------------------
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
  err = loss(model, data)
end

function r2(params, data)
  model = build_constitutive_model(params...)
  y_true = Float64[]
  y_pred = Float64[]

  if data isa Vector{LoadingTest}
    quantity_selector = r -> r.σ
  elseif data isa Vector{HeatingTest}
    quantity_selector = r -> r.cv
  end

  for d in data
    exp_vals = quantity_selector(d)
    append!(y_true, exp_vals)
    
    if d isa LoadingTest
      sim_vals = simulate_experiment(model, d.θ, d.Δt, d.λ)
    elseif d isa HeatingTest
      sim_vals = simulate_experiment(model, d.θ)
    end
    append!(y_pred, sim_vals)
  end

  y_mean = mean(y_true)
  ss_res = sum(abs2, y_true .- y_pred)
  ss_tot = sum(abs2, y_true .- y_mean)
  return 1 - (ss_res / ss_tot)
end

function stats(params, data, names=map("",params))
  r(num) = round(num, sigdigits=2)
  n_params = length(params)
  n_data = sum(length, data)
  n_dof = n_data - n_params

  sse_val = loss(params, data)
  res_variance = sse_val / n_dof

  H = FiniteDiff.finite_difference_hessian(p -> loss(p, data), params)
  local cov_matrix
  try
    cov_matrix = 2*res_variance*inv(H)
  catch
    foreach((p, n) -> println("$(n) : $(r(p))"), params, names)
    println("Warning: Singular hessian matrix. Probably there are redundant parameters.")
    return
  end

  t_crit = quantile(TDist(n_dof), 0.975) # t-Student value
  std_errs = sqrt.(abs.(diag(cov_matrix)))
  ci_lower = params .- t_crit .* std_errs
  ci_upper = params .+ t_crit .* std_errs

  for i in 1:n_params
    println("$(names[i]) : $(r(params[i])) ± $(r(t_crit*std_errs[i]))")
    println("     Interval : [$(r(ci_lower[i])) , $(r(ci_upper[i]))]")
    sens = H[i,i] * params[i]^2 / sse_val
    println("     Sensitivity : $(r(sens))")
  end
  println("R2 : ", lpad(@sprintf("%.1f", 100*r2(params,data)), 8))
  return ci_lower, ci_upper
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


#------------------------------------------
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


#------------------------------------------
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
stats(sol_heat, heating_data, ["cv0", "γv"])

cv0, γv = sol_heat.u
model = build_constitutive_model(cv0, γv)
text1 = text("cv⁰ = " * @sprintf("%.0f", cv0) * " N/(m²·K)\n" *
             "γ̄   = " * @sprintf("%.2f", γv) * "\n" *
             "R2  = " * @sprintf("%.0f", 100*r2(sol_heat.u, heating_data)) * " %",
             8, :left)

# Plot the solution
p = plot()
plot_experiment!(model, heating_data[1])
annotate!((0.05, 0.75), text1, relative=true)
display(p);


#------------------------------------------
# Reference characterization
#------------------------------------------
function reference_characterization(data)
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
sol_mech = reference_characterization(subset_T20)
l_mech, u_mech = stats(sol_mech.u, subset_T20, ["C10", "C20", "C30", "μ1", "p1"])
model  = build_constitutive_model(sol_mech.u...)
lmodel = build_constitutive_model(l_mech...)
umodel = build_constitutive_model(u_mech...)
r2(sol_mech.u, subset_T20)
text2 = text(@sprintf("R2 = %.1f %%", 100*r2(sol_mech.u, subset_T20)), 8, :left)

function plot_reference_config!(model, data, labelfn=_->"")
  σ_values = simulate_experiment(model, data.θ, data.Δt, data.λ)
  label = labelfn(data)
  plot!(data.λ, [σ_values, data.σ], label=[label ""], xlabel="Stretch [-]", ylabel="Stress [Pa]", typ=[:path :scatter], lw=2, mswidth=0, color_palette=colors2)
end
function plot_reference_config!(model, l, u, data)
  σ_values = simulate_experiment(model, data.θ, data.Δt, data.λ)
  σ_lower = simulate_experiment(l, data.θ, data.Δt, data.λ)
  σ_upper = simulate_experiment(u, data.θ, data.Δt, data.λ)
  plot!(data.λ, σ_values, label="Model", xlabel="Stretch [-]", ylabel="Stress [Pa]", lw=2, color_palette=colors4)
  plot!(data.λ, σ_lower,  label="Model ± STD", style=:dash, lw=1, color_palette=colors4)
  plot!(data.λ, σ_upper,  label="", style=:dash, lw=1, color_palette=colors4)
  plot!(data.λ, data.σ,   label="Experiment", typ=:scatter, mswidth=0, color_palette=colors4)
end

p = plot(title="20ºC, 0.1/s, 300%", titlefontsize=10)
experim = filter(r -> r.v≈0.1 && r.λ_max≈4.0, subset_T20)[1]
plot_reference_config!(model, lmodel, umodel, experim)
annotate!((0.05, 0.5), text2, relative=true)
display(p);

p = plot(title="20ºC, 0.1/s", titlefontsize=10)
for e in filter(r -> r.v ≈ 0.1, subset_T20)
  plot_reference_config!(model, e, stretch_label)
end
annotate!((0.05, 0.5), text2, relative=true)
display(p);

p = plot(title="20ºC, 300%", titlefontsize=10)
for e in filter(r -> r.λ_max ≈ 4.0, subset_T20)
  plot_reference_config!(model, e, vel_label)
end
annotate!((0.05, 0.5), text2, relative=true)
display(p);


#------------------------------------------
# Visco-elastic characterization
#------------------------------------------
function mechanical_characterization(data)
  #    [   C1,     C2,     C3,      μ1     p1,    γel,   γvis] #,   δel,  δvis]
  p0 = [  3e4,   -2e2,    3e0,   5.0e4,   0.0,    1.0,    1.0] #,   0.1,   0.1]  # Initial seed
  lb = [2.0e4, -2.0e3, -2.0e1,   1.0e4,  -5.0,  -10.0,  -10.0] #,  -1.0,  -1.0]  # Minimum search limits
  ub = [2.0e5,  2.0e3,  2.0e1,   1.0e5,   5.0,  100.0,  100.0] #,   1.0,   1.0]  # Maximum search limits
  opt_func = OptimizationFunction(loss)   # AutoFiniteDiff() is needed for gradient-based search algorithms
  opt_prob = OptimizationProblem(opt_func, p0, data, lb=lb, ub=ub)
  solve(opt_prob, ParticleSwarm(lower=lb, upper=ub, n_particles=100), maxiters=1000, maxtime=1000.0)  # ECA (Evolutionary Centers Algorithm), NelderMead, LBFGS
end

function plot_experiment!(model, data::LoadingTest, labelfn=d->"")
  σ_values = simulate_experiment(model, data.θ, data.Δt, data.λ)
  label = labelfn(data)
  plot!(data.λ, [σ_values, data.σ], label=[label ""], xlabel="Stretch [-]", ylabel="Stress [Pa]", typ=[:path :scatter], lw=2, mswidth=0, color_palette=colors2)
end

sol_mech = mechanical_characterization(mechanical_data)
stats(sol_mech.u, mechanical_data, ["C10", "C20", "C30", "μ1", "p1", "γel", "γvis"])#, "δel", "δvis"])
@show sol_mech.stats

C1, C2, C3, μ1, p1, γel, γvis = sol_mech.u
model = build_constitutive_model(sol_mech.u...)
text2 = text(" γ̂el = " * @sprintf("%.1f\n", γel) *
             "γ̂vis = " * @sprintf("%.1f\n", γvis) *
             "  R2 = " * @sprintf("%.1f %%", 100*r2(sol_mech.u, mechanical_data)),
             8, :left)

subset = filter(r -> (r.v ≈ 0.1 && r.λ_max ≈ 2 && r.θ > -10+K0), mechanical_data)
sort!(subset, by = r -> r.θ)
p = plot(title="0.1/s, 100%", titlefontsize=10)
for e ∈ subset
  plot_experiment!(model, e, temp_label)
end
plot!([], [], label="Experiment", color=:black, typ=:scatter, wswidth=0)
plot!([], [], label="Model",      color=:black, lw=2)
annotate!((0.05, 0.5), text2, relative=true)
display(p);

subset = filter(r -> (r.v ≈ 0.1 && r.λ_max ≈ 4 && r.θ > -10+K0), mechanical_data)
sort!(subset, by = r -> r.θ)
p = plot(title="0.1/s, 300%", titlefontsize=10)
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
