##
## Plots for the qualitative representation of experimental data
##
## Data and definitions
using Plots
using Printf

default(lw=2)
default(mswidth=0)
default(size=(400,300))
default(label=false)
default(palette=:seaborn_colorblind)

include("ExperimentsData.jl")

set_1_heat = load_data(abspath(@__DIR__, "data/set 1 calorimetry.csv"), CalorimetryTest)
set_2_cycle = load_data(abspath(@__DIR__, "data/set 2 loading.csv"), LoadingTest)
set_3_creep = load_data(abspath(@__DIR__, "data/set 3 creep.csv"), CreepTest)


## Calorimetry test
exp_1 = set_1_heat[1]
step = 3
p1 = plot(xlabel="Temperature [ºK]", ylabel="Specific heat [kJ/m³/ºk]")
scatter!(exp_1.θ[begin:step:end], exp_1.cv[begin:step:end] ./ 1e3)
display(p1);
savefig(p1, abspath(@__DIR__, "../article/figures/qualitative_experim_calorimetry.pdf"))


## One-cycle loading-unloading tests
p2 = plot(xlabel="Stretch [-]", ylabel="Stress [kPa]")
for θ ∈ [20, 40, 60] .+ 273.15
  v = 0.05
  λ = 4.0
  exp_2 = getfirst(r -> r.θ ≈ θ && r.v ≈ v && r.λ_max ≈ λ, set_2_cycle)
  label = @sprintf("%2.0fºC, %.2f/s", θ-273.15, v)
  scatter!(p2, exp_2.λ, exp_2.σ ./ 1e3; label)
end
display(p2);
savefig(p2, abspath(@__DIR__, "../article/figures/qualitative_experim_one_cycle.pdf"))


## Creep tests
p3 = plot(xlabel="Time [h]", ylabel="Stress [kPa]", ylims=[0,85])
t0 = 0
for λ ∈ [1.25, 2, 3, 4]
  exp_3 = getfirst(r -> r.λ_max ≈ λ, set_3_creep)
  label = @sprintf("%3.0f%%", (λ-1)*100)
  scatter!(exp_3.t ./ 3600 .+ (t0 += 0.5), exp_3.σ ./ 1e3; label)
end
display(p3);
savefig(p3, abspath(@__DIR__, "../article/figures/qualitative_experim_creep.pdf"))


## Dielectrical tests

