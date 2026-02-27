
using Plots, Printf
import Plots: mm

# See the available palettes in:
# https://juliagraphics.github.io/ColorSchemes.jl/stable/catalogue/
#
# Some selected palettes:
# - :default 
# - :seaborn_colorblind 
# - :seaborn_dark
# - :tableau_traffic

the_palette = palette(:seaborn_colorblind)
default(titlefontsize=10)
default(palette = the_palette)

const colors2 = mapreduce(c -> [c,c], vcat, the_palette)
const colors3 = mapreduce(c -> [c,c,c], vcat, the_palette)
const colors4 = mapreduce(c -> [c,c,c,c], vcat, the_palette)
const diverging_rb = palette([reverse(palette(:blues,50))...; palette(:OrRd,50)...])

const c1 = the_palette[1]

temp_label(data) = @sprintf("%2.0fºC", data.θ-K0)
vel_label(data) = @sprintf("%.2f/s", data.v)
stretch_label(data) = @sprintf("%.0f%%", 100*(data.λ_max-1))

function plot_experiment_legend!()
  plot!([], [], label="Experiment", color=:black, typ=:scatter, wswidth=0)
  plot!([], [], label="Model",      color=:black, lw=2)
end

function plot_experiment!(model, data::CalorimetryTest)
  cv_values = evaluate_cv(model, data.θ)
  plot!(data.θ.-K0, [cv_values, data.cv], label=["Model" "Experiment"], typ=[:path :scatter], lw=2, mswidth=0)
end

function plot_experiment!(model, data::LoadingTest, labelfn=d->"")
  σ_values = evaluate_stress(model, data.Δt, data.θ, data.λ)
  label = labelfn(data)
  plot!(data.λ, [σ_values, data.σ]./1e3, label=[label ""], typ=[:path :scatter], lw=2, mswidth=0, color_palette=colors2)
end

function plot_experiment!(model, data::CreepTest, labelfn=d->"")
  λ = fill(data.λ_max, size(data.t))
  σ_values = evaluate_stress(model, data.Δt, data.θ, λ)
  label = labelfn(data)
  plot!(data.t, [σ_values, data.σ]./1e3, label=[label ""], typ=[:path :scatter], lw=2, mswidth=0, color_palette=colors2)
end

function plot_experiment!(model, data::QuasiStaticTest, labelfn=d->"")
  σ_values = evaluate_stress(model, data.θ, data.λ)
  label = labelfn(data)
  plot!(data.λ, [σ_values, data.σ]./1e3, label=[label ""], typ=[:path :scatter], lw=2, mswidth=0, color_palette=colors2)
end

function plot_confidence_bands!(model, random_models, data)
  for rand_model in random_models
    σ_sim = evaluate_stress(rand_model, data.Δt, data.θ, data.λ)
    plot!(p, data.λ, σ_sim./1e3, color=c1, alpha=0.05, lw=1, label="")
  end

  σ_opt = evaluate_stress(model, data.Δt, data.θ, data.λ)
  plot!(p, data.λ, σ_opt./1e3, color=c1, lw=2, label="Model")
  
  scatter!(p, data.λ, data.σ./1e3, label="Experiment", color=:black, markerstrokewidth=0)
end

function plot_thermal_laws(x, law, title)
  f, df, ddf = derivatives(law)
  g = θ -> -θ*ddf(θ)
  funcs = [f, df, ddf, g]
  titles = title * " " .* ["f(θ)", "∂f(θ)", "∂∂f(θ)", "-θ·∂∂f(θ)"]
  p = map((f, t) -> plot(x.-K0, f.(x), title=t, lab="", lw=2, left_margin=5mm), funcs, titles)
  plot(p...; layout=@layout([a;b;c;d]), size=(600,800))
end
