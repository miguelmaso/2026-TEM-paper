
max_value(data::MechanicalTest) = data.σ_max

max_value(data::ThermalTest) = data.cv_max

max_value(data::ThermoDielectricData) = maximum(data.ϵ)

function loss(model::PhysicalModel, data::ExperimentData)
  y_data, y_pred = experiment_prediction(model, data)
  m = max_value(data)
  s2 = zero(eltype(y_pred))
  @inbounds for i in eachindex(y_data)
      s2 += abs2((y_pred[i] - y_data[i]) / m)
  end
  return (s2 / length(y_data)) * data.weight
end

function loss(model::PhysicalModel, data::Vector{<:ExperimentData})
  sum(d -> loss(model, d), data) / sum(d -> d.weight, data)
end

function loss(model_builder, params, data)
  model = model_builder(params...)
  loss(model, data)
end

function parallel_loss(model_builder, data)
  P -> begin
    score = zeros(size(P,1))
    Threads.@threads for i in axes(P,1)
      score[i] = loss(model_builder, P[i,:], data)
    end
    return score
  end
end

function experiment_prediction(model::PhysicalModel, data::LoadingTest)
  y_true = data.σ
  y_pred = evaluate_stress(model, data.Δt, data.θ, data.λ)
  return y_true, y_pred
end

function experiment_prediction(model::PhysicalModel, data::CoupledTest)
  if sum(data.σ0) == 0  # Default to standard simulation
    y_true = data.σ
    y_pred = evaluate_stress(model, data.Δt, data.θ, data.V, data.λ)
    return y_true, y_pred
  else   # Compute the stress increment due to voltage
    y_true = data.σ .- data.σ0
    y_V = evaluate_stress(model, data.Δt, data.θ, data.V, data.λ)
    y_0 = evaluate_stress(model, data.Δt, data.θ, 0.0   , data.λ)
    y_pred = y_V .- y_0
    return y_true, y_pred
  end
end

function experiment_prediction(model::PhysicalModel, data::CreepTest)
  y_true = data.σ
  x_data = fill(data.λ_max, size(data.t))
  y_pred = evaluate_stress(model, data.Δt, data.θ, x_data)
  return y_true, y_pred
end

function experiment_prediction(model::PhysicalModel, data::QuasiStaticTest)
  y_true = data.σ
  y_pred = evaluate_stress(model, data.θ, data.λ)
  return y_true, y_pred
end

function experiment_prediction(model::PhysicalModel, data::CalorimetryTest)
  y_true = data.cv
  y_pred = evaluate_cv(model, data.θ)
  return y_true, y_pred
end

function experiment_prediction(model::PhysicalModel, data::ThermoDielectricData)
  y_true = data.ϵ
  y_pred = evaluate_epsilon(model, data.θ)
  return y_true, y_pred
end

function experiment_prediction(model::PhysicalModel, data::Vector{<:ExperimentData})
  y = map(d -> experiment_prediction(model, d), data)
  return vcat(first.(y)...), vcat(last.(y)...)
end

function r_squared(model::PhysicalModel, data)
  y_true, y_pred = experiment_prediction(model, data)
  y_mean = mean(y_true)
  ss_res = sum(abs2, y_true .- y_pred)
  ss_tot = sum(abs2, y_true .- y_mean)
  return 1 - (ss_res / ss_tot)
end

function covariance_matrix(model_builder, params, data)
  n_params = length(params)
  n_data = npoints(data)
  n_dof = n_data - n_params

  sse_val = loss(model_builder, params, data)
  res_variance = sse_val / n_dof
  
  # Compute the covariance matrix
  H = FiniteDiff.finite_difference_hessian(p -> loss(model_builder, p, data), params)
  local cov_matrix
  try
    cov_matrix = 2*res_variance*pinv(H)
  catch
    println("⚠️ Singular hessian matrix. Probably there are redundant parameters.")
    return
  end
  cov_matrix, H
end

function stats(model_builder, params, data, names=map("",params); io::IO=stdout)
  model = model_builder(params...)
  n_dof = npoints(data) - length(params)
  sse_val = loss(model, data)
  cov_matrix, H = covariance_matrix(model_builder, params, data)

  t_crit = quantile(TDist(n_dof), 0.975) # t-Student value
  std_errs = sqrt.(abs.(diag(cov_matrix)))
  ci_lower = params .- t_crit .* std_errs
  ci_upper = params .+ t_crit .* std_errs

  for i in eachindex(params)
    abs_e = t_crit * std_errs[i]
    rel_e = abs(abs_e / params[i])
    sens = H[i,i] * params[i]^2 / sse_val
    @printf(io, "%-5s | % 8.2g ± %7.2g (%5.1f%%) | %5.1f\n", names[i], params[i], abs_e, 100rel_e, sens)
  end
  return r_squared(model, data)
end

function covariance_uncertainty(model_builder, params, data, n_samples=100)
  cov_matrix, _ = covariance_matrix(model_builder, params, data)
  M = (cov_matrix + cov_matrix') / 2
  vals, vecs = eigen(M)

  threshold = maximum(abs.(vals)) * 1e-8  # Security threshold (adjustable)
  if any(vals .<= threshold)
    println("⚠️  Repairing covariance matrix:")
    println("   Original eigenvalues: ", round.(vals, sigdigits=3))
    println("   New eigenvalues:      ", round.(vals_clean, sigdigits=3))
    
    vals_clean = max.(vals, threshold) # We must enforce positivity of the eigenvalues
    M_reconstructed = vecs * Diagonal(vals_clean) * vecs'
    M = Symmetric(M_reconstructed)
  end
  
  mv_param_dist = MvNormal(params, M)
  rand(mv_param_dist, n_samples) # Population of n samples of parameters sets
end

function bootstrap_uncertainity(model_builder, params, data, n_samples=20)
  # Estimating standard error (assuming gaussian noise)
  base_model = model_builder(params...)
  y_true, y_pred = experiment_prediction(base_model, data)
  residuals = y_pred .- y_true
  sigma_err = std(residuals)
  println("Estimated noise on data: ", sigma_err)

  bootstrapped_params = Vector{Vector{Float64}}()
  
  println("Starting bootstrapping ($n_samples samples)")
  Threads.@threads for _ in 1:n_samples
    # A. Generating synthetic data (theroetical curve + noise)
    synthetic_data = map(d -> begin
      T = typeof(d)
      y_true, y_clean = experiment_prediction(base_model, d)
      y_noise = y_clean .+ randn(length(y_clean)) * sigma_err
      return T(d, y_noise)
    end, data)

    opt_func = OptimizationFunction((p, d) -> loss(model_builder, p, d))
    opt_prob = OptimizationProblem(opt_func, params, synthetic_data)
    sol = solve(opt_prob, NelderMead(), maxiters=100) # Low maxiter because we start close

    push!(bootstrapped_params, sol.u)
    print(".") # progress bar
  end
  println("\nBootstrapping completed")
  return hcat(bootstrapped_params...)
end
