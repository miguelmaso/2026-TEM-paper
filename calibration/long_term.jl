using Gridap
using HyperFEM
using Plots
using Printf

F_iso(λ) = TensorValue(λ, 0, 0, 0, λ^(-1/2), 0, 0, 0, λ^(-1/2))
C_iso(λ) = F_iso(λ)' * F_iso(λ)
β(N,λ) = sqrt(tr(C_iso(λ)) / 3 / N)

function energy(model, λ)
    Ψ = model()[1]
    map(λ_values) do λ
        F = F_iso(λ)
        return Ψ(F)
    end
end

function piola(model, λ_values)
    P = model()[2]
    map(λ_values) do λ
        F = F_iso(λ)
        return P(F)[1,1]
    end
end

μ = 9e5
N = 50.5

model = EightChain(μ=μ, N=N)
λ_values = [1:0.1:7...]
β_values = β.(N, λ_values)
E_values = energy(model, λ_values)
P_values = piola(model, λ_values)
p1 = plot(λ_values, E_values, label="", title="Energy")
p2 = plot(λ_values, P_values, label="", title="Stress")
p3 = plot(λ_values, β_values, label="", title="Beta")

# display(plot(p1, p2, p3, layout=@layout([a; b; c])))
display(p2)
