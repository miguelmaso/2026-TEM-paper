#
# Calibration of VHB 4905 polymer.
# Data and libraries from:
# https://doi.org/10.1016/j.ijnonlinmec.2019.103263 - Liao, Mokarram et al., 2022, On thermo-viscoelastic experimental characterization and numerical...
# https://doi.org/10.1002/zamm.201400110 - Dippel et al., 2015, Thermo-mechanical couplings in elastomers - experiments and modelling
# https://doi.org/10.1016/j.ijsolstr.2022.111523 - Alkhoury at al., 2022, Experiments and modeling of the thermo-mechanically coupled behavior of VHB
# https://docs.sciml.ai/Optimization/stable/#Citation - Kumar, 2023, Optimization.jl: A unified optimization package
# https://automeris.io - Ankit Rohatgi, WebPlot Digitizer, v5.2
#
## Packages and definitions

using Revise
using Plots, Printf
using HyperFEM, HyperFEM.ComputationalModels.EvolutionFunctions
using StaticArrays
using Metaheuristics
using KernelAbstractions
using Optimization, OptimizationOptimJL, OptimizationMetaheuristics, Optim
using ParallelParticleSwarms
using LinearAlgebra, FiniteDiff, Distributions
using JLD2

import Optimization: solve  # Avoid potential conflict with Gridap.solve

include("ConstitutiveModelling.jl")
include("ExperimentsData.jl")
include("ObjectiveFunctions.jl")
include("ExperimentsPlots.jl")


## Load experimental data

set_1_cal   = load_data(abspath(@__DIR__, "data/set 1 calorimetry.csv"), CalorimetryTest)
set_2_load  = load_data(abspath(@__DIR__, "data/set 2 loading.csv"), LoadingTest)
set_3_creep = load_data(abspath(@__DIR__, "data/set 3 creep.csv"), CreepTest)
set_4_quasi = load_data(abspath(@__DIR__, "data/set 4 quasi-static.csv"), QuasiStaticTest)
set_5_load  = load_data(abspath(@__DIR__, "data/set 5 loading.csv"), LoadingTest)
set_6_creep = load_data(abspath(@__DIR__, "data/set 6 creep.csv"), CreepTest)
set_7_elec  = load_data(abspath(@__DIR__, "data/set 7 dielectric.csv"), DielectricTest)
set_8_coupl = load_data(abspath(@__DIR__, "data/set 8 coupled.csv"), CoupledTest)

foreach(r -> r.weight = 0.1, set_3_creep)

println(set_1_cal)
println(set_2_load)
println(set_3_creep)
println(set_4_quasi)
println(set_5_load)
println(set_6_creep)
println(set_8_coupl)


## Step 1: Thermal characterization

build_heat(cv0, γv, κr) = ThermalVolumetric(cv0=cv0, θr=θr, α=αr, κr=κr, κ=1.0, γ=γv)

pn = ["cv0","γv",  "κr" ]  # Parameter names
p0 = [1.0e6, 0.5, 1.0e9 ]  # Initial seed
lb = [ 10.0, 0.0, 1.0e8 ]  # Minimum search limits
ub = [1.0e8, 1.0, 1.0e10]  # Maximum search limits

opt_func = OptimizationFunction((p, d) -> loss(build_heat, p, d))
opt_prob = OptimizationProblem(opt_func, p0, set_1_cal, lb=lb, ub=ub)
opt_heat = solve(opt_prob, Optim.ParticleSwarm(lower=lb, upper=ub, n_particles=100), maxiters=1000, maxtime=60)
sol_heat = opt_heat.u

model = build_heat(sol_heat...)
r2 = stats(build_heat, sol_heat, set_1_cal, pn)
text_r2 = text(@sprintf("R² = %.0f %%", 100*r2), 12, :left)

p = plot(xlabel="T [ºC]", ylabel="cv [J/m³·ºK]")
plot_experiment!(model, set_1_cal[1])
annotate!((0.05, 0.8), text_r2, relative=true)
display(p);
savefig(p, abspath(@__DIR__, "..//article//figures//volumetric_characterization.pdf"));


## Step 2: Hyperelastic characterization

build_longterm(μ, N) = EightChain(μ=μ, N=N)
pn = [  "μ",   "N"]  # Parameter names
p0 = [  1e4,  30.0]  # Initial seed
lb = [  1e3,  30.0]  # Lower search limits
ub = [  1e5,  80.0]  # Upper search limits

build_longterm(C1, C2, C3) = Yeoh3D(λ=0.0, C10=C1, C20=C2, C30=C3)
pn = ["C10",  "C20",  "C30"]  # Parameter names
p0 = [  3e4,   -2e2,    3e0]  # Initial seed
lb = [1.0e3, -2.0e3,  0.0e0]  # Minimum search limits
ub = [2.0e5,  2.0e3,  2.0e2]  # Maximum search limits

build_longterm(μ1, μ2, α1, α2) = NonlinearMooneyRivlin3D(λ=0.0, μ1=μ1, μ2=μ2, α1=α1, α2=α2)
pn = [ "μ1", "μ2", "α1", "α2"]  # Parameter names
p0 = [  1e4,  1e4,  0.8,  0.8]  # Initial seed
lb = [  1e2,  1e3,  0.5,  0.5]  # Lower search limits
ub = [  1e5,  1e5,  3.0,  2.0]  # Upper search limits

opt_func = OptimizationFunction((p,d) -> loss(build_longterm, p, d))
opt_prob = OptimizationProblem(opt_func, p0, set_4_quasi, lb=lb, ub=ub)
opt_long = solve(opt_prob, ParticleSwarm(lower=lb, upper=ub, n_particles=1000), maxiters=1000, maxtime=60)
opt_prob = OptimizationProblem(opt_func, opt_long.u, set_4_quasi)
opt_long = solve(opt_prob, Optim.NelderMead(), maxiters=100, maxtime=30)
sol_long = opt_long.u

model = build_longterm(sol_long...)
r2 = stats(build_longterm, sol_long, set_4_quasi, pn)
text_par = text(join(map((n,v) -> @sprintf("%s=%.2g",n,v), pn, sol_long), "\n"), 12, :left)
text_r2 = text(@sprintf("R² = %.1f %%", 100*r2), 12, :left)

p = plot(xlabel="Stretch [-]", ylabel="Stress [KPa]")
plot_experiment!(model, getfirst(r -> r.θ ≈ θr, set_4_quasi))
annotate!((0.05, 0.8), text_r2, relative=true)
display(p);
savefig(p, abspath(@__DIR__, "..//article//figures//long_term_characterization.pdf"));


## Auxiliary plot for comparison of hyperelastic models
if false
  experim = getfirst(r -> r.θ ≈ θr, set_4_quasi)
  s_yeoh = evaluate_stress(yeoh_model, experim.λ)
  s_8chain = evaluate_stress(arruda_model, experim.λ)
  p = plot(xlabel="Stretch [-]", ylabel="Stress [KPa]")
  plot!(experim.λ, [s_yeoh, s_8chain].*1e-3, label=["Yeoh" "8-chain"])
  scatter!(experim.λ, experim.σ * 1e-3, label="Experiment")
  display(p);
  savefig(p, abspath(@__DIR__, "..//article//figures//long_term_comparison.pdf"));
end


## Step 3: Viscoelastic characterization

build_branch(μ, t) = ViscousIncompressible(IncompressibleNeoHookean3D(λ=0.0, μ=μ), τ=exp10(t))
build_branches(p...) = map(splat(build_branch), Iterators.partition(p,2))
build_visco(p...) = GeneralizedMaxwell(build_longterm(sol_long...), build_branches(p...)...)
n_branches = 3
pn = reduce(vcat, ["μ$i", "t$i"] for i in 1:n_branches)  # Parameter names
p0 = reduce(vcat, [  1e4,   1.0] for _ in 1:n_branches)  # Initial seed
lb = reduce(vcat, [  1e3,  -1.0] for _ in 1:n_branches)  # Lower search limits
ub = reduce(vcat, [  1e5,   4.0] for _ in 1:n_branches)  # Upper search limits

set_2_ref = filter(r -> r.θ ≈ θr, set_2_load)
set_3_ref = filter(r -> r.θ ≈ θr, set_3_creep)
set_23_ref = [set_2_ref; set_3_ref]

opt_func = OptimizationFunction((p,d) -> loss(build_visco, p, d))
opt_prob = OptimizationProblem(opt_func, p0, set_23_ref, lb=lb, ub=ub)
opt_visco = solve(opt_prob, ParticleSwarm(lower=lb, upper=ub, n_particles=100), maxiters=1000, maxtime=600)
opt_prob = OptimizationProblem(opt_func, opt_visco.u, set_23_ref)
opt_visco = solve(opt_prob, Optim.NelderMead(), maxiters=100, maxtime=30)
sol_visco = opt_visco.u

model = build_visco(sol_visco...)
r2 = stats(build_visco, sol_visco, set_2_ref, pn)
text_param = text(join(map((n,v) -> @sprintf("%s=%.2g",n,v), pn, sol_visco), "\n"), 12, :left)

subset = filter(r -> r.v ≈ 0.1, set_2_ref)
p = plot_experiments(model, subset, temp_vel_label, stretch_label, "Stretch [-]", "Stress [KPa]")
display(p);
savefig(p, abspath(@__DIR__, "..//article//figures//viscous_characterization_vel.pdf"));

subset = filter(r -> r.λ_max ≈ 4.0, set_2_ref)
p = plot_experiments(model, subset, temp_stretch_label, vel_label, "Stretch [-]", "Stress [KPa]")
display(p);
savefig(p, abspath(@__DIR__, "..//article//figures//viscous_characterization_stretch.pdf"));


# rand_params = covariance_uncertainty(build_visco, sol_visco, set_23_ref)
# rand_models = map(splat(build_visco), eachcol(rand_params))

# p = plot(title="20ºC, 0.1/s, 300%\n95% confidence bands", xlabel="Stretch [-]", ylabel="Stress [KPa]")
# experim = getfirst(r -> r.v≈0.1 && r.λ_max≈4.0, set_2_ref)
# plot_confidence_bands!(model, rand_models, experim, alpha=0.1)
# display(p);


## Step 4: Thermo-mechanical characterization

build_g1(γ) = EntropicElasticityLaw(θr=θr, γ=γ)
build_g2(γ) = NonlinearMeltingLaw(θr=θr, θM=150+273.15, γ=γ)
build_g3(μ, γ, δ) = NonlinearSofteningLaw(θr=θr, θt=μ, γ=γ, δ=δ)

build_TM(γe, μv, γv, δv) = ThermoMech_Bonet(build_heat(sol_heat...), build_visco(sol_visco...), build_g2(γe), build_g3(μv, γv, δv))
pn = @MArray ["γel", "θvis", "γvis", "δvis"]  # Parameter names
p0 = @MArray [  0.5,   270.,    5.0,    0.2]  # Initial seed
lb = @MArray [  0.1,   250.,    4.0,    0.0]  # Minimum search limits
ub = @MArray [  2.0,   300.,   10.0,    0.5]  # Maximum search limits

set_2_θ = filter(r -> r.θ > 0+K0, set_2_load)

# opt_func = parallel_loss(build_TM, set_2_θ)
# options = Options(iterations=100, parallel_evaluation=true);
# opt_therm = Metaheuristics.optimize(opt_func, [lb ub], PSO(;options))
# sol_therm = opt_therm.best_sol.x


opt_func = (p, data) -> loss(build_TM, p, data)
opt_prob = OptimizationProblem(opt_func, p0, set_2_θ; lb, ub)
opt_therm = solve(opt_prob, ParticleSwarm(lower=lb, upper=ub, n_particles=100), maxiters=1000, maxtime=120) # ParallelPSOKernel(100, backend=KernelAbstractions.CPU())
opt_prob  = OptimizationProblem(opt_func, opt_therm.u, set_2_θ)
opt_therm = solve(opt_prob, NelderMead(), maxiters=100, maxtime=30) 
sol_therm = opt_therm.u


model = build_TM(sol_therm...)
stats(build_TM, sol_therm, set_2_θ, pn)

subset = sort(filter(r -> (r.v ≈ 0.1 && r.λ_max ≈ 4 && r.θ > 273), set_2_θ), by = r -> r.θ)
p = plot_experiments(model, subset, vel_stretch_label, temp_label, "Stretch [-]", "Stress [KPa]")
display(p);


## Plot thermal laws
display(plot_thermal_laws(0:5:500, model.thermo.law, "Volumetric law"));
display(plot_thermal_laws(0:5:500, model.lawel, "Long term law"));
display(plot_thermal_laws(0:5:500, model.lawvis, "Viscous law"));


## Plot specific heat map

v = 0.05
θ_vals_cv  = 1:50:2.06θr
λ_vals_cv  = 1:0.5:8.0
cv_vals_cv = @. evaluate_cv(model, θ_vals_cv, λ_vals_cv', v)
cv_vals_cv = replace(cv_vals_cv, NaN=>missing)
cv_lim = maximum(abs.(skipmissing(cv_vals_cv)))
cv_vals_cv = clamp.(cv_vals_cv, -cv_lim, cv_lim)
levels = range(-cv_lim, cv_lim, length=100)
p = plot(title="Specific heat under isochoric stretch, v=$v/s", xlabel="Stretch [-]", ylabel="θ/θᵣ [-]", rightmargin=8mm, framestyle=:grid)
contourf!(λ_vals_cv, θ_vals_cv./θr, cv_vals_cv, color=diverging_rb, lw=0, lc=:transparent, clims=(-cv_lim, cv_lim), levels=levels)
plot!([1.02, 3.98, 3.98, 1.02, 1.02], ([-20, -20, 80, 80, -20].+K0)./θr, color=:black, lw=2, label="")
display(p);


## Step 5: Thermo-electrical characterization

build_g_elec(γ, θM) = NonlinearMeltingLaw(θr=θr, γ=γ, θM=θM)
build_elec(ϵr) = IdealDielectric(ε = ϵr*ϵ0)
build_TE(ϵr, γ, θM) = ThermoElectroModel(build_elec(ϵr), build_g_elec(γ, θM))

pn = [  "ϵr",  "γ",   "θM"]
p0 = [   4.0,  1.0,  200.0]
lb = [   1.0,  0.0,  100.0]
ub = [  10.0,  5.0, 1000.0]

data_elec = ThermoDielectricData(set_7_elec, 1e1)  # Training at frequency f=1e1

opt_func = OptimizationFunction((p,d) -> loss(build_TE, p, d))
opt_prob = OptimizationProblem(opt_func, p0, data_elec, lb=lb, ub=ub)
opt_elec = solve(opt_prob, ParticleSwarm(lower=lb, upper=ub, n_particles=100), maxiters=5000, maxtime=60)
sol_elec = opt_elec.u

stats(build_TE, sol_elec, data_elec, pn)


## Step 6: Thermo-electro-mechanical validation

model = ThermoElectroMech_Bonet(build_heat(sol_heat[1], sol_heat[2], 0.0), build_TE(sol_elec...), build_visco(sol_visco...), el=build_g2(sol_therm[1]), vis=build_g3(sol_therm[2], sol_therm[3], sol_therm[4]))

p = plot()
for e in set_8_coupl
  plot_experiment!(model, e, temp_voltage_label)
end
display(p);
savefig(p, abspath(@__DIR__, "../article/figures/fully_coupled_experiments.pdf"))


## Save/load variables

# @load "res/3_branches.jld2" sol_heat sol_long sol_visco sol_therm

@load "res/sol_heat.jld2" sol_heat
@load "res/sol_long.jld2" sol_long
@load "res/sol_3_br.jld2" sol_visco
@load "res/sol_therm.jld2" sol_therm
@load "res/sol_elec.jld2" sol_elec

