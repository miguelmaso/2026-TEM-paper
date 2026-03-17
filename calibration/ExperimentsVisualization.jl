##
## Plots for the qualitative representation of experimental data
##
## Data and definitions
using Plots
using Printf

pgfplotsx()
default(lw=2)
default(mscolor=:transparent)
default(size=(400,300))
default(label=false)
default(palette=:seaborn_colorblind)

include("ExperimentsData.jl")

set_1_heat = load_data(abspath(@__DIR__, "data/set 1 calorimetry.csv"), CalorimetryTest)
set_2_cycle = load_data(abspath(@__DIR__, "data/set 2 loading.csv"), LoadingTest)
set_3_creep = load_data(abspath(@__DIR__, "data/set 3 creep.csv"), CreepTest)
set_7_dielec = load_data(abspath(@__DIR__, "data/set 7 dielectrical.csv"), DielectricalTest)


## Calorimetry test
exp_1 = set_1_heat[1]
step = 3
p1 = plot(xlabel="Temperature [K]", ylabel="Specific heat [kJ/m³/K]")
scatter!(exp_1.θ[begin:step:end], exp_1.cv[begin:step:end] ./ 1e3)
display(p1);
# savefig(p1, abspath(@__DIR__, "../article/figures/qualitative_experim_calorimetry.pdf"))


## One-cycle loading-unloading tests
p2 = plot(xlabel="Stretch [-]", ylabel="Stress [kPa]", legend=:topleft)
for θ ∈ [20, 40, 60] .+ 273.15
  v = 0.05
  λ = 4.0
  exp_2 = getfirst(r -> r.θ ≈ θ && r.v ≈ v && r.λ_max ≈ λ, set_2_cycle)
  label = @sprintf("%2.0fºC, %.2f/s", θ-273.15, v)
  scatter!(p2, exp_2.λ, exp_2.σ ./ 1e3; label)
end
display(p2);
# savefig(p2, abspath(@__DIR__, "../article/figures/qualitative_experim_one_cycle.pdf"))


## Creep tests
p3 = plot(xlabel="Time [h]", ylabel="Stress [kPa]", ylims=[0,85], legend=:topleft)
t0 = 0
for λ ∈ [1.25, 2, 3, 4]
  exp_3 = getfirst(r -> r.λ_max ≈ λ, set_3_creep)
  label = @sprintf("%3.0f%%", (λ-1)*100)
  scatter!(p3, exp_3.t ./ 3600 .+ (global t0 += 0.5), exp_3.σ ./ 1e3; label)
end
display(p3);
# savefig(p3, abspath(@__DIR__, "../article/figures/qualitative_experim_creep.pdf"))


## Dielectrical tests
p7 = plot(xlabel="Frequency [Hz]", ylabel="Dielectrical constant [-]", xaxis=:log10, ylims=[3,5], legend=:bottomleft)
for θ ∈ [0, 20, 40, 60, 80] .+ 273.15
  exp_7 = getfirst(r -> r.θ ≈ θ, set_7_dielec)
  label = @sprintf("%2.0fºC", θ-273.15)
  scatter!(p7, exp_7.f, exp_7.ϵ; label)
end
display(p7);
savefig(p7, abspath(@__DIR__, "../article/figures/qualitative_experim_dielectrical.pdf"))
