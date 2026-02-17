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
using Serialization

include("ConstitutiveModelling.jl")
include("ExperimentsData.jl")
include("ObjectiveFunctions.jl")
include("ExperimentsPlots.jl")

##-----------------------------------------
# Constitutive models
# -----------------------------------------

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

heating_data = read_data(joinpath(@__DIR__, "data/Dippel 2015.csv"), HeatingTest)
mechanical_data = read_data(joinpath(@__DIR__, "data/Liao_Mokarram 2022.csv"), LoadingTest)
foreach(r -> r.θ < K0-10 && (r.weight *= 0.5), mechanical_data)
foreach(r -> r.θ > K0+70 && (r.weight *= 0.5), mechanical_data)
foreach(r -> r.λ_max < 4 && (r.weight *= 2.0), mechanical_data)
filter!(r -> (r.θ > -10+K0), mechanical_data)
println(heating_data)
println(mechanical_data)


##-----------------------------------------
# Thermal characterization
#------------------------------------------
build_heat(cv0, γv) = yeoh_1_branch_exp(1.0e4, 1.0e2, 1.0e0, 4.0e4, 1.0, cv0, γv, 0.5, 0.5, 0.0, 0.0)

function heat_characterization(data)
  #    [  cv0,  γv]
  p0 = [1.0e3, 0.5]  # Initial seed
  lb = [ 10.0, 0.0]  # Minimum search limits
  ub = [1.0e5, 1.0]  # Maximum search limits
  opt_func = OptimizationFunction((p, d) -> loss(build_heat, p, d))
  opt_prob = OptimizationProblem(opt_func, p0, data, lb=lb, ub=ub)
  solve(opt_prob, Optim.ParticleSwarm(lower=lb, upper=ub, n_particles=100), maxiters=1000, maxtime=60.0)
end

sol_heat = heat_characterization(heating_data)
model = build_heat(sol_heat.u...)

stats(build_heat, sol_heat, heating_data, ["cv0", "γv"])
text1 = text(@sprintf("R² = %.0f %%", 100*r_squared(model, heating_data)), 8, :left)

# Plot the solution
p = plot(title="Volumetric characterization"; label_λσ...)
plot_experiment!(model, heating_data[1])
annotate!((0.05, 0.75), text1, relative=true)
display(p);


##-----------------------------------------
# Reference characterization
#------------------------------------------
yeoh_model(C1, C2, C3, μ1, p1) = yeoh_1_branch_exp(C1, C2, C3, μ1, p1, 1283.88, 0.78, 0.0, 0.0, 0.0, 0.0)

function viscoelastic_characterization(data)
  #    [   C1,     C2,     C3,      μ1     p1]
  p0 = [  3e4,   -2e2,    3e0,   5.0e4,   0.0]  # Initial seed
  lb = [1.0e4, -2.0e3,  1.0e0,   1.0e4,  -5.0]  # Minimum search limits
  ub = [2.0e5,  2.0e3,  2.0e2,   1.0e5,   5.0]  # Maximum search limits
  opt_func = OptimizationFunction((p,d) -> loss(yeoh_model, p, d))

  opt_prob = OptimizationProblem(opt_func, p0, data, lb=lb, ub=ub)
  sol = solve(opt_prob, ParticleSwarm(lower=lb, upper=ub, n_particles=100), maxiters=1000, maxtime=60.0)

  opt_prob = OptimizationProblem(opt_func, sol.u, data)
  sol = solve(opt_prob, NelderMead())
end

subset_T20 = filter(r -> r.θ ≈ 20+K0, mechanical_data)
sol_mech = viscoelastic_characterization(subset_T20)
model = yeoh_model(sol_mech.u...)

stats(yeoh_model, sol_mech.u, subset_T20, ["C10", "C20", "C30", "μ1", "p1"])
text2 = text(@sprintf("R² = %.1f %%", 100*r_squared(model, subset_T20)), 8, :left)

rand_params = covariance_uncertainity(yeoh_model, sol_mech.u, subset_T20)
rand_models = map(splat(yeoh_model), eachcol(rand_params))

p = plot(title="20ºC, 0.1/s, 300%\n95% confidence bands"; label_λσ...)
experim = getfirst(r -> r.v≈0.1 && r.λ_max≈4.0, subset_T20)
plot_confidence_bands!(model, rand_models, experim)
annotate!((0.05, 0.75), text2, relative=true)
display(p);

p = plot(title="20ºC, 0.1/s"; label_λσ...)
for e in filter(r -> r.v ≈ 0.1, subset_T20)
  plot_experiment!(model, e, stretch_label)
end
annotate!((0.05, 0.75), text2, relative=true)
display(p);

p = plot(title="20ºC, 300%"; label_λσ...)
for e in filter(r -> r.λ_max ≈ 4.0, subset_T20)
  plot_experiment!(model, e, vel_label)
end
annotate!((0.05, 0.75), text2, relative=true)
display(p);


##-----------------------------------------
# Visco-elastic characterization
#------------------------------------------
build_therm(Mel, Mvis) = yeoh_1_branch_trign(sol_mech.u..., sol_heat.u..., Mel, Mvis)
build_therm(γel, γvis, δel, δvis) = yeoh_1_branch_exp(sol_mech.u..., sol_heat.u..., γel, γvis, δel, δvis)
build_therm(e1, e2, e3, v1, v2, v3) = yeoh_1_branch_poly(sol_mech.u..., sol_heat.u..., e1, e2, e3, v1, v2, v3)

# pn = [ "θMel", "θMvis"]  # Parameter names
# p0 = [  1.2θr,   1.2θr]  # Initial seed
# lb = [     θr,      θr]  # Minimum search limits
# ub = [  5.0θr,   5.0θr]  # Maximum search limits

# pn = [ "γel", "γvis", "δel", "δvis"]  # Parameter names
# p0 = [  10.0,   10.0,   0.1,    0.1]  # Initial seed
# lb = [   0.0,    0.0,   0.0,    0.0]  # Minimum search limits
# ub = [  50.0,   50.0,   1.0,    1.0]  # Maximum search limits

pn = [    "e1",    "e2",    "e3",    "v1",    "v2",    "v3"]  # Parameter names
p0 = [ -1.0e-5,  1.0e-3, -1.0e-2, -2.0e-4,  1.0e-2, -2.0e-2]  # Initial seed
lb = [ -5.0e-5, -1.0e-3, -5.0e-2, -5.0e-4,  0.0e-2, -1.0e-1]  # Minimum search limits
ub = [  1.0e-5,  2.0e-3,  0.0e-2,  0.0e-4,  2.0e-2,  0.0e-0]  # Maximum search limits

opt_func = OptimizationFunction((p, d) -> loss(build_therm, p, d), AutoFiniteDiff())
opt_prob = OptimizationProblem(opt_func, p0, mechanical_data, lb=lb, ub=ub)
sol_therm = solve(opt_prob, ParticleSwarm(lower=lb, upper=ub, n_particles=100), maxiters=1000, maxtime=5*60)

model = build_therm(sol_therm.u...)

stats(build_therm, sol_therm.u, mechanical_data, pn)
text3 = text(@sprintf("R² = %.1f %%", 100*r_squared(model,mechanical_data)), 8, :left)

subset = filter(r -> (r.v ≈ 0.1 && r.λ_max ≈ 2 && r.θ > -10+K0), mechanical_data)
sort!(subset, by = r -> r.θ)
p = plot(title="0.1/s, 100%")
for e ∈ subset
  plot_experiment!(model, e, temp_label)
end
plot!([], [], label="Experiment", color=:black, typ=:scatter, wswidth=0)
plot!([], [], label="Model",      color=:black, lw=2)
annotate!((0.05, 0.65), text3, relative=true)
display(p);

subset = filter(r -> (r.v ≈ 0.1 && r.λ_max ≈ 4 && r.θ > -10+K0), mechanical_data)
sort!(subset, by = r -> r.θ)
p = plot(title="0.1/s, 300%")
for e ∈ subset
  plot_experiment!(model, e, temp_label)
end
plot!([], [], label="Experiment", color=:black, typ=:scatter, wswidth=0)
plot!([], [], label="Model",      color=:black, lw=2)
annotate!((0.05, 0.62), text3, relative=true)
display(p);


##---------------------------
# Specific heat plot
# ---------------------------
v = 0.03
θ_vals_cv  = 0.1:10:2θr
λ_vals_cv  = 1:0.1:5.0
cv_vals_cv = @. cv_single_step_stretch(model, λ_vals_cv', θ_vals_cv, v)
cv_vals_cv = replace(cv_vals_cv, NaN=>missing)
cv_lim = 0.1*maximum(abs.(skipmissing(cv_vals_cv)))
cv_vals_cv = clamp.(cv_vals_cv, -cv_lim, cv_lim)
p = plot(title="Specific heat under isochoric stretch, v=$v/s", xlabel="Stretch [-]", ylabel="θ/θR [-]", rightmargin=8mm, framestyle=:grid)
contourf!(λ_vals_cv, θ_vals_cv./θr, cv_vals_cv, color=diverging_rb, clims=(-cv_lim, cv_lim), lw=0)
plot!([1.02, 3.98, 3.98, 1.02, 1.02], ([-20, -20, 80, 80, -20].+K0)./θr, color=:black, lw=2, label="")
display(p);
xlims!(1,4)
ylims!(253/θr,354/θr)
display(p);

display(plot_thermal_laws([-20:0.5:80...].+K0, model.gd, "Elastic-deviatoric"))
display(plot_thermal_laws([-20:0.5:80...].+K0, model.gvis, "Viscous-deviatoric"))

ηR = HyperFEM.entropy(model)[1]
Ψe = model.mechano.longterm()[1]
Ψv = model.mechano.branches()[1]
fR = derivatives(model.gv)[3]
fe = derivatives(model.gd)[3]
fv = derivatives(model.gvis)[3]
∂∂Ψ = model()[5]

update_time_step!(model, 1/0.03)
A1 = VectorValue(I3..., 0.0)
F1 = F_iso(1.0)
F2 = F_iso(2.0)
@assert ηR(F2) ≈ sol_heat[1]/sol_heat[2]
r2(x) = round(x; sigdigits=2)
@printf "cv = θ·θR·∂∂f·ηR   - θ·∂∂f·Ψe - θ·∂∂f·Ψα\n"
@printf "   =     %.2g·%.0f" θr*θr*fR(θr) r2(ηR(F2))
@printf " %.2g·%.0f"   -θr*fe(θr) r2(Ψe(F2))
@printf " %.2g·%.0f\n" -θr*fv(θr) r2(Ψv(F2,F1,A1))
@printf "   =     %.0f      %.0f      %.0f\n" θr*θr*fR(θr)*ηR(F2) -θr*fe(θr)*Ψe(F2) -θr*fv(θr)*Ψv(F2,F1,A1)
@printf "   =    %.0f\n" -θr*∂∂Ψ(F2,θr,F1,A1)


##---------------------------
# Save/load veriables
# ---------------------------
# serialize(joinpath(@__DIR__, "polynomial.bin"), (sol_heat, sol_mech, sol_therm))
# (sol_heat, sol_mech, sol_therm) = deserialize(joinpath(@__DIR__, "polynomial.bin"))


##---------------------------
# Dummy plot
# ---------------------------
p = plot()
cons_model = yeoh_1_branch_exp(20.e3, -630., 20., 54.e3, log(12.2), 1280.0, 0.78, 1.0, 3.0, 0.0, 0.0)
# cons_model = neo_hookean_1_branch(1.3e4, 4.4e4, 2.2, 1280.0, 0.78, 0.0, 0.0, 0.0, 0.0)
λ_max = 4.0
t_max = (λ_max-1) / 0.1
Δt = t_max / 20
λ_values = map(1 + (λ_max-1)*triangular(t_max), 0:Δt:1.6*t_max)
for T in [0.0, 20.0, 40.0]
  σ_values = simulate_experiment(cons_model, T+K0, Δt, λ_values) / 1e3
  plot!(λ_values, σ_values, label="T = $T ºC", xlabel="Stretch [-]", ylabel="Stress [kPa]", lw=2)
end
# display(p);
