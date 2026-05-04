
using Plots, Printf
import Plots: mm

pgfplotsx() # Enable LaTeX fonts for labels
            # This backend is slow, use GR() (the default) for a faster rendering
            # PgfPlots backend has several compatibility issues and depends on the LaTeX installation

# See the available palettes in:
# https://juliagraphics.github.io/ColorSchemes.jl/stable/catalogue/
#
# Some selected palettes:
# - :default 
# - :seaborn_colorblind 
# - :seaborn_dark
# - :tableau_traffic

the_palette = palette(:seaborn_colorblind)
fontsize = 12

default(legendfontsize = fontsize)
default(tickfontsize = fontsize)
default(labelfontsize = fontsize)
default(titlefontsize = fontsize)
default(palette = the_palette)
default(linewidth = 2)
default(mscolor = :transparent)  # mswidth is not recognized by pgfplotsx, setting transparent color is a workaround
default(legend = :topleft)       # pgfplotsx tends to place the legend outside the plot

const colors2 = mapreduce(c -> [c,c], vcat, the_palette)
const colors3 = mapreduce(c -> [c,c,c], vcat, the_palette)
const colors4 = mapreduce(c -> [c,c,c,c], vcat, the_palette)
const diverging_rb = cgrad([reverse(palette(:blues,50))...; palette(:OrRd,50)...], categorical=true)

const c1 = the_palette[1]

vel_label(data) = @sprintf("%.2f/s", data.v)
temp_label(data) = @sprintf("%2.0fºC", data.θ-K0)
stretch_label(data) = @sprintf("%3.0f%%", 100*(data.λ_max-1))
temp_vel_label(data) = temp_label(data) * ", " * vel_label(data)
vel_stretch_label(data) = vel_label(data) * ", " * stretch_label(data)
temp_stretch_label(data) = temp_label(data) * ", " * stretch_label(data)
temp_vel_stretch_label(data) = temp_label(data) * ", " * vel_label(data) * ", " * stretch_label(data)

creep_time_offset = Ref(0.0)

function plot_experiment_legend!(; color=:black)
  plot!([], [], label="Experiment", color=color, typ=:scatter)
  plot!([], [], label="Model",      color=color)
end

function plot_experiment!(model, data::CalorimetryTest)
  cv_values = evaluate_cv(model, data.θ)
  plot!(data.θ.-K0, [cv_values, data.cv], label=["Model" "Experiment"], typ=[:path :scatter])
end

function plot_experiment!(model, data::LoadingTest, labelfn=d->"")
  σ_values = evaluate_stress(model, data.Δt, data.θ, data.λ)
  label = labelfn(data)
  plot!(data.λ, [σ_values, data.σ]./1e3, label=[label ""], typ=[:path :scatter], color_palette=colors2)
end

function plot_experiment!(model, data::CoupledTest, labelfn=d->"")
  σ_values = evaluate_stress(model, data.Δt, data.θ, data.V, data.λ)
  label = labelfn(data)
  plot!(data.λ, [σ_values, data.σ]./1e3, label=[label ""], typ=[:path :scatter], color_palette=colors2)
end

function plot_experiment!(model, data::CreepTest, labelfn=d->"")
  λ = fill(data.λ_max, size(data.t))
  σ_values = evaluate_stress(model, data.Δt, data.θ, λ)
  label = labelfn(data)
  plot!(data.t./3600 .+ creep_time_offset[], [σ_values, data.σ]./1e3, label=[label ""], typ=[:path :scatter], color_palette=colors2)
  creep_time_offset[] += 0.5
end

function plot_experiment!(model, data::QuasiStaticTest)
  σ_values = evaluate_stress(model, data.θ, data.λ)
  plot!(data.λ, [σ_values, data.σ]./1e3, label=["Model" "Experiment"], typ=[:path :scatter], color_palette=colors2)
end

function plot_confidence_bands!(model, random_models, data; alpha=0.05)
  for rand_model in random_models
    σ_sim = evaluate_stress(rand_model, data.Δt, data.θ, data.λ)
    plot!(p, data.λ, σ_sim./1e3, color=c1, alpha=alpha, lw=1, label="")
  end

  σ_opt = evaluate_stress(model, data.Δt, data.θ, data.λ)
  plot!(p, data.λ, σ_opt./1e3, color=c1, lw=2, label="Model")
  
  scatter!(p, data.λ, data.σ./1e3, label="Experiment", color=:black, markerstrokewidth=0)
end

function plot_thermal_laws(x, law, title)
  f, df, ddf = law()
  g = θ -> -θ*ddf(θ)
  funcs = [f, df, ddf, g]
  titles = title * " " .* ["f(θ)", "∂f(θ)", "∂∂f(θ)", "-θ·∂∂f(θ)"]
  p = map((f, t) -> plot(x.-K0, f.(x), title=t, lab="", lw=2, left_margin=5mm), funcs, titles)
  plot(p...; layout=@layout([a;b;c;d]), size=(600,800))
end

function plot_experiments(model, data, titlefn, labelfn, xlabel, ylabel)
  p = plot(; title=titlefn(data[1]), xlabel, ylabel)
  creep_time_offset[] = 0.0
  for e ∈ data
    plot_experiment!(model, e, labelfn)
  end
  plot_experiment_legend!()
  return p
end

function annotate_r2!(value, pos)
  text_r2 = text(@sprintf("R² = %.0f %%", 100*value), fontsize, :left)
  annotate!((0.02, pos), text_r2, relative=true)
end
