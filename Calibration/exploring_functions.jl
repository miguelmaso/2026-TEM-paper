using Plots; default(titlefontsize=14, legend=false)

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
