using Plots; default(titlefontsize=14, legend=false)
using SpecialFunctions

θr = 1.0
θM = 1.4θr

g(θ) = θ/θr * sin(2π*θ/θM)
G(θ) = 1/2/π * θM/θr * (1 - cos(2π*θ/θM))
H(θ) = 1/2/π * θM/θr * (θ - θM/2/π * sin(2π*θ/θM))
f(θ) = 1+(H(θr) - H(θ)) / (H(θM) - H(θr))
∂f(θ) = -G(θ) / (H(θM) - H(θr))
∂∂f(θ) = g(θ) / θ / (H(θM) - H(θr))

θ_range = 0:0.02:θM
p1 = plot(θ_range, θ -> f(θ), title="f")
p2 = plot(θ_range, θ -> ∂f(θ), title="f'")
p3 = plot(θ_range, θ -> θ*∂∂f(θ), title="θf''")
plot!(p1, [θr], [f(θr)], typ=:scatter, msw=0, txt=text(" f(θr)=1", :left))
plot(p1, p2, p3; layout=@layout[a;b;c], size=(600,600))



struct LogisticLaw <: ThermalLaw
  θr::Float64
  μ::Float64
  σ::Float64
end

function derivatives(law::LogisticLaw)
  @unpack θr, μ, σ = law
  z(x) = (log(x) - μ) / σ
  std_pdf(x) = 1/σ/sqrt(2 * π) * exp(-z(x)^2 / 2)
  std_cdf(x) = 0.5 * (1 + erf(z(x) / sqrt(2)))
  ξR = 1 / (1-std_cdf(θr))
  f(θ) = ξR * (1-std_cdf(θ))
  ∂f(θ) = -ξR / θ * std_pdf(θ)
  ∂∂f(θ) = ξR / θ^2 * std_pdf(θ) * (1 + z(θ)/σ)
  return (f, ∂f, ∂∂f)
end

law = LogisticLaw(293.15, 5.7, 0.5)
f, ∂f, ∂∂f = derivatives(law)
θ_range = 0:1:2θr
p1 = plot(θ_range, θ -> f(θ), title="f")
p2 = plot(θ_range, θ -> ∂f(θ), title="f'")
p3 = plot(θ_range, θ -> ∂∂f(θ), title="f''")
p4 = plot(θ_range, θ -> θ*∂∂f(θ), title="θf''")
plot!(p1, [θr], [f(θr)], typ=:scatter, msw=0, txt=text(" f(θr)=1", :left))
plot(p1, p2, p3, p4; layout=@layout[a;b;c;d], size=(600,600))
