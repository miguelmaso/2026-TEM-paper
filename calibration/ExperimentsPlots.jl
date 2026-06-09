
using Plots, Printf
using LaTeXStrings
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
# default(extra_kwargs = Dict(:subplot => Dict("legend style" => "{cells={anchor=west}, at={(0.02,0.98)}, anchor=north west}")))

const colors2 = mapreduce(c -> [c,c], vcat, the_palette)
const colors3 = mapreduce(c -> [c,c,c], vcat, the_palette)
const colors4 = mapreduce(c -> [c,c,c,c], vcat, the_palette)
const diverging_rb = cgrad([reverse(palette(:blues,50))...; palette(:OrRd,50)...], categorical=true)

vel_label(data) = @sprintf("%.2f/s", data.v)
temp_label(data) = @sprintf("%2.0fºC", data.θ-K0)
stretch_label(data) = @sprintf("%3.0f%%", 100*(data.λ_max-1))
voltage_label(data) = @sprintf("%4dV", data.V)
temp_vel_label(data) = temp_label(data) * ", " * vel_label(data)
vel_stretch_label(data) = vel_label(data) * ", " * stretch_label(data)
temp_stretch_label(data) = temp_label(data) * ", " * stretch_label(data)
temp_voltage_label(data) = temp_label(data) * ", " * voltage_label(data)
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
    plot!(data.λ, σ_sim./1e3, color=1, alpha=alpha, lw=1, label="")
  end

  σ_opt = evaluate_stress(model, data.Δt, data.θ, data.λ)
  plot!(data.λ, σ_opt./1e3, color=1, lw=2, label="Model")
  
  scatter!(data.λ, data.σ./1e3, label="Experiment", color=:black, markerstrokewidth=0)
end

function plot_thermal_laws(x, law)
  f, df, ddf = law()
  θr = law.θr
  ζr = 1/(df(θr))
  ξr = 1/(ddf(θr)*ζr*θr)
  G = θ -> ζr*df(θ)
  g = θ -> θ*ξr*ζr*ddf(θ)
  fmt = (v) -> @sprintf("%.1f", v)
  yt1 = optimize_ticks(extrema(replace(f.(x), NaN=>0.0))...; k_max=6)[1]
  yt2 = optimize_ticks(extrema(replace(G.(x), NaN=>0.0))...; k_max=6)[1]
  yt3 = optimize_ticks(extrema(replace(g.(x), NaN=>0.0))...; k_max=6)[1]
  p1 = plot(x./θr, f.(x), ylabel="f = Ψ/Ψᵣ",   color=1, label=false, formatter=fmt, xticks=0:0.5:2, yticks=yt1, xaxis=false, bottom_margin=-12mm)
  p2 = plot(x./θr, G.(x), ylabel="G = η/ηᵣ",   color=2, label=false, formatter=fmt, xticks=0:0.5:2, yticks=yt2, xaxis=false, bottom_margin=-12mm)
  p3 = plot(x./θr, g.(x), ylabel="g = cᵥ/cᵥ⁰", color=3, label=false, formatter=fmt, xticks=0:0.5:2, yticks=yt3, xlabel="θ/θᵣ",)
  scatter!(p1, [1], [1], color=1, label=false)
  scatter!(p2, [1], [1], color=2, label=false)
  scatter!(p3, [1], [1], color=3, label=false)
  plot(p1, p2, p3, layout=grid(3,1, heights=[0.3, 0.3, 0.4]), link=:x)
end

function plot_thermal_laws(laws::Vector{ThermalLaw}, labels::Vector{String})
  x = 0.01θr:5:2.0θr
  fmt = (v) -> @sprintf("%.1f", v)
  p1 = plot(formatter=fmt, ylabel=L"f = \Psi/\Psi_R", xaxis=false, bottom_margin=-12mm)
  p2 = plot(formatter=fmt, ylabel=L"G = \eta/\eta_R", xaxis=false, bottom_margin=-12mm)
  p3 = plot(formatter=fmt, ylabel=L"g = c_v/c_v^0", xlabel = L"\theta/\theta_R", xticks=0:0.5:1.5)
  for (law, label) in zip(laws, labels)
    f, df, ddf = law()
    θr = law.θr
    ζr = 1/(df(θr))
    ξr = 1/(ddf(θr)*ζr*θr)
    G = θ -> ζr*df(θ)
    g = θ -> θ*ξr*ζr*ddf(θ)
    plot!(p1, x./θr, f.(x), label=label)
    plot!(p2, x./θr, G.(x), label=false)
    plot!(p3, x./θr, g.(x), label=false)
  end
  scatter!(p1, [1], [1], color=:black, label=false)
  scatter!(p2, [1], [1], color=:black, label=false)
  scatter!(p3, [1], [1], color=:black, label=false)
  ylims!(p1, 0.0, Inf)
  plot(p1, p2, p3, layout=grid(3,1, heights=[0.3, 0.3, 0.4]), link=:x)
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
