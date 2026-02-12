
const colors2 = mapreduce(c -> [c,c], vcat, palette(:default))
const colors3 = mapreduce(c -> [c,c,c], vcat, palette(:default))
const colors4 = mapreduce(c -> [c,c,c,c], vcat, palette(:default))

const temp_label = data -> @sprintf("%2.0fºC", data.θ-K0)
const vel_label = data -> @sprintf("%.2f/s", data.v)
const stretch_label = data -> @sprintf("%.0f%%", 100*(data.λ_max-1))

const label_λσ = (; xlabel="Stretch [-]", ylabel="Stress [Pa]")
default(titlefontsize=10)

function plot_experiment!(model, data::HeatingTest)
  cv_values = simulate_experiment(model, data.θ)
  plot!(data.θ.-K0, [cv_values, data.cv], label=["Model" "Experiment"], xlabel="T [ºC]", ylabel="cv [J/(kg·ºK)]", lw=2, mark=[:none :circle], markerstrokewidth=0)
end

function plot_experiment!(model, data::LoadingTest, labelfn=d->"")
  σ_values = simulate_experiment(model, data.θ, data.Δt, data.λ)
  label = labelfn(data)
  plot!(data.λ, [σ_values, data.σ], label=[label ""], typ=[:path :scatter], lw=2, mswidth=0, color_palette=colors2)
end

function plot_confidence_bands!(model, random_models, data)
  for rand_model in random_models
    σ_sim = simulate_experiment(rand_model, data.θ, data.Δt, data.λ)
    plot!(p, data.λ, σ_sim, color=:blue, alpha=0.05, lw=1, label="")
  end

  σ_opt = simulate_experiment(model, data.θ, data.Δt, data.λ)
  plot!(p, data.λ, σ_opt, color=:blue, lw=2, label="Model")
  
  scatter!(p, data.λ, data.σ, label="Experiment", color=:black, markerstrokewidth=0)
end
