#------------------------------------------
#------------------------------------------
#
# Calibration of VHB 4905 polymer.
# Data and libraries from:
# https://doi.org/10.1016/j.ijnonlinmec.2019.103263 - Liao, Mokarram et al., 2022, On thermo-viscoelastic experimental characterization and numerical modelling of VHB polymer
# https://doi.org/10.1002/zamm.201400110 - Dippel et al., 2015, Thermo-mechanical couplings in elastomers - experiments and modelling
# https://doi.org/10.1016/j.ijsolstr.2022.111523 - Alkhoury at al., 2022, Experiments and modeling of the thermo-mechanically coupled behavior of VHB
# https://docs.sciml.ai/Optimization/stable/#Citation - Kumar, 2023, Optimization.jl: A unified optimization package
# https://automeris.io - Ankit Rohatgi, WebPlot Digitizer, v5.2
#
#------------------------------------------
#------------------------------------------

using Plots, Printf
using HyperFEM, HyperFEM.ComputationalModels.EvolutionFunctions
using Optimization, OptimizationOptimJL, OptimizationMetaheuristics, Optim
using LinearAlgebra, FiniteDiff, Distributions
using Serialization

include("ConstitutiveModelling.jl")
include("ExperimentsData.jl")
include("ObjectiveFunctions.jl")
include("ExperimentsPlots.jl")

##-----------------------------------------
# Constitutive models
# -----------------------------------------

function yeoh_1_branch_logist(C1, C2, C3, μ1, p1, cv0, γv, μel, σel, μvis, σvis)
  long_term = Yeoh3D(λ=0.0, C10=C1, C20=C2, C30=C3)
  branch_1 = ViscousIncompressible(IncompressibleNeoHookean3D(λ=0., μ=μ1), τ=exp(p1))
  visco_elasto = GeneralizedMaxwell(long_term, branch_1)
  thermal_model = ThermalModel(Cv=cv0, θr=θr, α=αr, κ=1.0)
  func_v = VolumetricLaw(θr, γv)
  func_el = LogisticLaw(θr, μel, σel)
  func_vis = LogisticLaw(θr, μvis, σvis)
  return ThermoMech_Bonet(thermal_model, visco_elasto, func_v, func_el, func_vis)
end

function yeoh_1_branch_bonet(C1, C2, C3, μ1, p1, cv0, γv, θM, γel, γvis)
  long_term = Yeoh3D(λ=0.0, C10=C1, C20=C2, C30=C3)
  branch_1 = ViscousIncompressible(IncompressibleNeoHookean3D(λ=0., μ=μ1), τ=exp(p1))
  visco_elasto = GeneralizedMaxwell(long_term, branch_1)
  thermal_model = ThermalModel(Cv=cv0, θr=θr, α=αr, κ=1.0)
  func_v = VolumetricLaw(θr, γv)
  func_el = EntropicMeltingLaw(θr, θM, γel)
  func_vis = EntropicMeltingLaw(θr, θM, γvis)
  return ThermoMech_Bonet(thermal_model, visco_elasto, func_v, func_el, func_vis)
end

function yeoh_1_branch_poly(C1, C2, C3, μ1, p1, cv0, γv, e1, e2, e3, v1, v2, v3)
  long_term = Yeoh3D(λ=0.0, C10=C1, C20=C2, C30=C3)
  branch_1 = ViscousIncompressible(IncompressibleNeoHookean3D(λ=0., μ=μ1), τ=exp(p1))
  visco_elasto = GeneralizedMaxwell(long_term, branch_1)
  thermal_model = ThermalModel(Cv=cv0, θr=θr, α=αr, κ=1.0)
  func_v = VolumetricLaw(θr, γv)
  func_el = PolynomialLaw(θr, e1, e2, e3)
  func_vis = PolynomialLaw(θr, v1, v2, v3)
  return ThermoMech_Bonet(thermal_model, visco_elasto, func_v, func_el, func_vis)
end

function yeoh_1_branch_trign(C1, C2, C3, μ1, p1, cv0, γv, Mel, Mvis)
  long_term = Yeoh3D(λ=0.0, C10=C1, C20=C2, C30=C3)
  branch_1 = ViscousIncompressible(IncompressibleNeoHookean3D(λ=0., μ=μ1), τ=exp(p1))
  visco_elasto = GeneralizedMaxwell(long_term, branch_1)
  thermal_model = ThermalModel(Cv=cv0, θr=θr, α=αr, κ=1.0)
  func_v = VolumetricLaw(θr, γv)
  func_el = TrigonometricLaw(θr, Mel)
  func_vis = TrigonometricLaw(θr, Mvis)
  return ThermoMech_Bonet(thermal_model, visco_elasto, func_v, func_el, func_vis)
end

function yeoh_1_branch_exp(C1, C2, C3, μ1, p1, cv0, γv, γel, γvis, δel, δvis)
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

set_1 = load_data(abspath(@__DIR__, "data/set 1 calorimetry.csv"), CalorimetryTest)
set_2 = load_data(abspath(@__DIR__, "data/set 2 loading.csv"), LoadingTest)
set_4 = load_data(abspath(@__DIR__, "data/set 4 quasi-static.csv"), QuasiStaticTest)
set_5 = load_data(abspath(@__DIR__, "data/set 5 loading.csv"), LoadingTest)
set_6 = load_data(abspath(@__DIR__, "data/set 6 creep.csv"), CreepTest)

println(set_1)
println(set_2)
println(set_4)
println(set_5)
println(set_6)


##-----------------------------------------
# Step 1: Thermal characterization
#------------------------------------------
build_heat(cv0, γv) = ThermoMech_Bonet(ThermalModel(Cv=cv0, θr=θr, α=αr, κ=1.0), NeoHookean3D(λ=0.0, μ=1e3), γv=γv, γd=0.5)

pn = ["cv0","γv"]  # Parameter names
p0 = [1.0e6, 0.5]  # Initial seed
lb = [ 10.0, 0.0]  # Minimum search limits
ub = [1.0e8, 1.0]  # Maximum search limits

opt_func = OptimizationFunction((p, d) -> loss(build_heat, p, d))
opt_prob = OptimizationProblem(opt_func, p0, set_1, lb=lb, ub=ub)
sol_heat = solve(opt_prob, Optim.ParticleSwarm(lower=lb, upper=ub, n_particles=100), maxiters=1000, maxtime=60)

model = build_heat(sol_heat.u...)
r2 = stats(build_heat, sol_heat, set_1, pn)
text1 = text(@sprintf("R² = %.0f %%", 100*r2), 8, :left)

# Plot the solution
p = plot(title="Volumetric characterization", xlabel="T [ºC]", ylabel="cv [J/m³·ºK]")
plot_experiment!(model, set_1[1])
annotate!((0.05, 0.75), text1, relative=true)
display(p);


##-----------------------------------------
# Step 2: Hyperelastic characterization
#------------------------------------------
build_longterm(C1, C2, C3) = Yeoh3D(λ=0.0, C10=C1, C20=C2, C30=C3)
pn = ["C10",  "C20",  "C30"]  # Parameter names
p0 = [  3e4,   -2e2,    3e0]  # Initial seed
lb = [1.0e3, -2.0e3,  0.0e0]  # Minimum search limits
ub = [2.0e5,  2.0e3,  2.0e2]  # Maximum search limits

build_longterm(μ, N) = EightChain(μ=μ, N=N)
pn = [  "μ",  "N"]  # Parameter names
p0 = [  1e4,  2e6]  # Initial seed
lb = [  1e3,  1e6]  # Lower search limits
ub = [  1e5, 1e10]  # Upper search limits

opt_func = OptimizationFunction((p,d) -> loss(build_longterm, p, d))
opt_prob = OptimizationProblem(opt_func, p0, set_4, lb=lb, ub=ub)
sol_long = solve(opt_prob, ParticleSwarm(lower=lb, upper=ub, n_particles=1000), maxiters=1000, maxtime=60)

model = build_longterm(sol_long.u...)
r2 = stats(build_longterm, sol_long.u, set_4, pn)
text2 = text(@sprintf("R² = %.1f %%", 100*r2), 8, :left)

p = plot(title="Long term characterization", xlabel="Stretch [-]", ylabel="Stress [KPa]")
plot_experiment!(model, getfirst(r -> r.θ ≈ θr, set_4))
annotate!((0.05, 0.7), text2, relative=true)
display(p);


##-----------------------------------------
# Step 2: Reference characterization
#------------------------------------------
yeoh_model(C1, C2, C3, μ1, p1) = yeoh_1_branch_exp(C1, C2, C3, μ1, p1, sol_heat.u..., 0.0, 0.0, 0.0, 0.0)

subset_T20 = filter(r -> r.θ ≈ 20+K0 && r.λ_max < 5.0, set_2)

pn = ["C10",  "C20",  "C30",    "μ1",  "p1"]
p0 = [  3e4,   -2e2,    3e0,   5.0e4,   0.0]  # Initial seed
lb = [1.0e4, -2.0e3,  1.0e0,   1.0e4,  -5.0]  # Minimum search limits
ub = [2.0e5,  2.0e3,  2.0e2,   1.0e5,   5.0]  # Maximum search limits

opt_func = OptimizationFunction((p,d) -> loss(yeoh_model, p, d))
opt_prob = OptimizationProblem(opt_func, p0, subset_T20, lb=lb, ub=ub)
sol_mech = solve(opt_prob, ParticleSwarm(lower=lb, upper=ub, n_particles=100), maxiters=1000, maxtime=60)

opt_prob = OptimizationProblem(opt_func, sol_mech.u, subset_T20)
sol_mech = solve(opt_prob, NelderMead())

model = yeoh_model(sol_mech.u...)

r2 = stats(yeoh_model, sol_mech.u, subset_T20, pn)
text2 = text(@sprintf("R² = %.1f %%", 100*r2), 8, :left)

rand_params = covariance_uncertainity(yeoh_model, sol_mech.u, subset_T20)
rand_models = map(splat(yeoh_model), eachcol(rand_params))

p = plot(title="20ºC, 0.1/s, 300%\n95% confidence bands", xlabel="Stretch [-]", ylabel="Stress [KPa]")
experim = getfirst(r -> r.v≈0.1 && r.λ_max≈4.0, subset_T20)
plot_confidence_bands!(model, rand_models, experim)
annotate!((0.05, 0.75), text2, relative=true)
display(p);

p = plot(title="20ºC, 0.1/s", xlabel="Stretch [-]", ylabel="Stress [KPa]")
for e in filter(r -> r.v ≈ 0.1, subset_T20)
  plot_experiment!(model, e, stretch_label)
end
plot_experiment_legend!()
annotate!((0.05, 0.68), text2, relative=true)
display(p);

p = plot(title="20ºC, 300%", xlabel="Stretch [-]", ylabel="Stress [KPa]")
for e in filter(r -> r.λ_max ≈ 4.0, subset_T20)
  plot_experiment!(model, e, vel_label)
end
plot_experiment_legend!()
annotate!((0.05, 0.68), text2, relative=true)
display(p);


##-----------------------------------------
# Visco-elastic characterization
#------------------------------------------
build_therm(Mel, Mvis) = yeoh_1_branch_trign(sol_mech.u..., sol_heat.u..., Mel, Mvis)
build_therm(θM, γel, γvis) = yeoh_1_branch_bonet(sol_mech.u..., sol_heat.u..., θM, γel, γvis)
# build_therm(γel, γvis, δel, δvis) = yeoh_1_branch_exp(sol_mech.u..., sol_heat.u..., γel, γvis, δel, δvis)
build_therm(μel, σel, μvis, σvis) = yeoh_1_branch_logist(sol_mech.u..., sol_heat.u..., μel, σel, μvis, σvis)
build_therm(e1, e2, e3, v1, v2, v3) = yeoh_1_branch_poly(sol_mech.u..., sol_heat.u..., e1, e2, e3, v1, v2, v3)

# pn = [ "θMel", "θMvis"]  # Parameter names
# p0 = [  1.2θr,   1.2θr]  # Initial seed
# lb = [     θr,      θr]  # Minimum search limits
# ub = [  5.0θr,   5.0θr]  # Maximum search limits

# pn = [  "θM", "γel", "γvis"]  # Parameter names
# p0 = [ 473.0,   0.5,    0.5]  # Initial seed
# lb = [ 373.0,   0.0,    0.0]  # Minimum search limits
# ub = [ 673.0,   1.0,    1.0]  # Maximum search limits

pn = [ "μel", "σel", "μvis", "σvis"]  # Parameter names
p0 = [ 273.0,   0.2,  273.0,    0.2]  # Initial seed
lb = [ 173.0,   1.1,  173.0,    1.1]  # Minimum search limits
ub = [ 800.0, 1.0e3,  800.0,  1.0e3]  # Maximum search limits

# pn = [ "γel", "γvis", "δel", "δvis"]  # Parameter names
# p0 = [  10.0,   10.0,   0.1,    0.1]  # Initial seed
# lb = [   0.0,    0.0,   0.0,    0.0]  # Minimum search limits
# ub = [  50.0,   50.0,   1.0,    1.0]  # Maximum search limits

# pn = [    "e1",    "e2",    "e3",    "v1",    "v2",    "v3"]  # Parameter names
# p0 = [  -1.0e1,   1.0e2,  -1.0e1,  -2.0e2,   1.0e3,  -1.0e1]  # Initial seed
# lb = [  -1.0e2,   0.0e0,  -1.0e2,  -1.0e3,   0.0e3,  -1.0e2]  # Minimum search limits
# ub = [   0.0e2,   2.0e2,   0.0e2,   0.0e3,   2.0e3,   0.0e2]  # Maximum search limits

opt_func = OptimizationFunction((p, d) -> loss(build_therm, p, d), AutoFiniteDiff())
opt_prob = OptimizationProblem(opt_func, p0, mechanical_data, lb=lb, ub=ub)
sol_therm = solve(opt_prob, ParticleSwarm(lower=lb, upper=ub, n_particles=200), maxiters=1000, maxtime=20*60)

model = build_therm(sol_therm.u...)

stats(build_therm, sol_therm.u, mechanical_data, pn)
text3 = text(@sprintf("R² = %.1f %%", 100*r_squared(model,mechanical_data)), 8, :left)

subset = filter(r -> (r.v ≈ 0.1 && r.λ_max ≈ 2 && r.θ > -10+K0), mechanical_data)
sort!(subset, by = r -> r.θ)
p = plot(title="0.1/s, 100%")
for e ∈ subset
  plot_experiment!(model, e, temp_label)
end
plot_experiment_legend!()
annotate!((0.05, 0.65), text3, relative=true)
display(p);

subset = filter(r -> (r.v ≈ 0.1 && r.λ_max ≈ 4 && r.θ > -10+K0), mechanical_data)
sort!(subset, by = r -> r.θ)
p = plot(title="0.1/s, 300%")
for e ∈ subset
  plot_experiment!(model, e, temp_label)
end
plot_experiment_legend!()
annotate!((0.05, 0.62), text3, relative=true)
display(p);


##---------------------------
# Specific heat plot
# ---------------------------
v = 0.03
θ_vals_cv  = 50:10:2θr
λ_vals_cv  = 1:0.1:5.0
cv_vals_cv = @. cv_single_step_stretch(model, λ_vals_cv', θ_vals_cv, v)
cv_vals_cv = replace(cv_vals_cv, NaN=>missing)
cv_lim = maximum(abs.(skipmissing(cv_vals_cv)))
cv_vals_cv = clamp.(cv_vals_cv, -cv_lim, cv_lim)
p = plot(title="Specific heat under isochoric stretch, v=$v/s", xlabel="Stretch [-]", ylabel="θ/θR [-]", rightmargin=8mm, framestyle=:grid)
contourf!(λ_vals_cv, θ_vals_cv./θr, cv_vals_cv, color=diverging_rb, clims=(-cv_lim, cv_lim), lw=0)
plot!([1.02, 3.98, 3.98, 1.02, 1.02], ([-20, -20, 80, 80, -20].+K0)./θr, color=:black, lw=2, label="")
display(p);

display(plot_thermal_laws([-20:0.5:80...].+K0, model.gd, "Elastic-deviatoric"))
display(plot_thermal_laws([-20:0.5:80...].+K0, model.gvis, "Viscous-deviatoric"))


##---------------------------
# Save/load veriables
# ---------------------------
# serialize(joinpath(@__DIR__, "logistic.bin"), (sol_heat, sol_mech, sol_therm))
# (sol_heat, sol_mech, sol_therm) = deserialize(joinpath(@__DIR__, "exponential.bin"))

