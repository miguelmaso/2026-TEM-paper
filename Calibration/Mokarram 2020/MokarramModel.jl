using ForwardDiff
using Parameters
using Plots
using Printf

import Plots:mm
const palette_1 = palette([:black, :red, :blue, :green])
const palette_2 = mapreduce(c -> [c,c], vcat, palette_1)
const θr = 20 + 273.15

include("../ExperimentsData.jl")
data = read_data(abspath(@__DIR__, "../data/Liao_Mokarram 2020.csv"), LoadingTest)

#region Definitions

## Types

abstract type ConstitutiveModel end
abstract type ElasticModel <: ConstitutiveModel end
abstract type ViscousModel <: ConstitutiveModel end
abstract type ViscoElasticModel <: ConstitutiveModel end

struct Carroll <: ElasticModel
    a::Real 
    b::Real
    c::Real
end

struct Yeoh <: ElasticModel
    μ::Real
end

struct NeoHooke <: ElasticModel
    μ::Real
end

struct ViscousBranch{T} <:ViscousModel where{T<:ElasticModel}
    e::T
    τ::Real
end

struct GeneralizedMaxwell{N} <: ViscoElasticModel
    longterm::ElasticModel
    branches::NTuple{N,ViscousBranch}
end

struct ThermoMechanicalModel{N} <: ConstitutiveModel
    mechano::GeneralizedMaxwell{N}
    thermal_laws::NTuple{N,Function}
end

## Thermal laws

function g1(θ, θr, c)
    exp((θr/(θr-40))^c - (θ/(θr-40))^c)
end

function g2(θ, θr, a, b)
    ((θr/θ)^(a*θr/θ) + b) / (1 + b)
end

## Free energy density, Ψ

function free_energy_density(model::Carroll, λ::Real)
    @unpack a, b, c = model
    a*(λ^2 + 2/λ) + b*(λ^2 + 2/λ)^4 + c*sqrt(2λ + 1/λ^2)
end

function free_energy_density(model::Yeoh, λ::Real)
    μ = model.μ
    μ * (λ^2 + 2/λ - 3)^3
end

function free_energy_density(model::NeoHooke, λ::Real)
    μ = model.μ
    μ * (λ^2 + 2/λ - 3)
end

function free_energy_density(model::ViscousBranch, λ::Real, λv::Real)
    free_energy_density(model.e, λ/λv)
end

function free_energy_density(model::GeneralizedMaxwell, λ::Real, A)
    Ψe = free_energy_density(model.longterm, λ)
    mapreduce((b,λv) -> free_energy_density(b,λ,λv), +, model.branches, A; init=Ψe)
end

function free_energy_density(model::ThermoMechanicalModel, λ::Real, θ::Real, A)
    Ψe = free_energy_density(model.mechano.longterm, λ)
    branches = model.mechano.branches
    laws = model.thermal_laws
    mapreduce((b,g,λv) -> g(θ)*free_energy_density(b,λ,λv), +, branches, laws, A; init=Ψe)
end

## Stress, P = ∂Ψ/∂F = ∂Ψ/∂λ

function stress(model::Carroll, λ::Real)
    @unpack a, b, c = model
    (2a + 8b*(2/λ + λ^2)^3 + c*(1 + 2λ^3)^(-1/2)) * (λ - 1/λ^2)
end

function stress(model::Yeoh, λ::Real)
    μ = model.μ
    6μ * (λ^2 + 2/λ - 3)^2 * (λ - 1/λ^2)
end

function stress(model::NeoHooke, λ::Real)
    μ = model.μ
    2μ * (λ - 1/λ^2)
end

function stress(model::ViscousBranch, λ::Real, λv::Real)
    stress(model.e, λ/λv) / λv  # P = ∂Ψ/∂λ = ∂Ψ/∂λe·∂λe/∂λ = Pe/λv
end

function stress(model::GeneralizedMaxwell, λ::Real, A)
    Pe = stress(model.longterm, λ)
    mapreduce((b,λv) -> stress(b,λ,λv), +, model.branches, A; init=Pe)
end

function stress(model::ThermoMechanicalModel, λ, θ, A)
    Pe = stress(model.mechano.longterm, λ)
    branches = model.mechano.branches
    laws = model.thermal_laws
    mapreduce((b,g,λv) -> g(θ)*stress(b,λ,λv), +, branches, laws, A; init=Pe)
end

## Specific heat, cv = -θ·∂²Ψ/∂θ²

function specific_heat(model::ThermoMechanicalModel, λ::Real, θ::Real, A)
    cv0 = 1280.0 * 720.0  # Cv0 * ρ
    branches = model.mechano.branches
    laws = model.thermal_laws
    deriv(f) = x -> ForwardDiff.derivative(f,x)
    d_laws = map(deriv, laws)
    dd_laws = map(deriv, d_laws)
    mapreduce((b,∂∂g,λv) -> -θ*∂∂g(θ)*free_energy_density(b,λ,λv), +, branches, dd_laws, A; init=cv0)
end

## Return mapping

function hessian(model::Yeoh, λ::Real)
    μ = model.μ
    I1 = λ^2 + 2/λ
    6μ * (4*(I1 - 3)*(λ - 1/λ^2)^2 + (I1 - 3)^2*(1 + 2/λ^3))
end

function hessian(model::NeoHooke, λ::Real)
    μ = model.μ
    2μ * (1 + 2/λ^3)
end

function hessian(model::ViscousBranch, λ::Real, λv::Real)
    -1 * hessian(model.e, λ/λv) * λ/λv^2  # H = ∂P/∂λv = ∂P/∂λe·∂λe/∂λv = He·(-λ/λv²)
end

function viscous_evolution(model::ViscousBranch, Δt::Real, λ::Real, λv_old::Real)
    η = model.τ * model.e.μ
    dλv(λv) = 2λ/(3η) * stress(model.e, λ/λv)
    R(λv) = λv - λv_old - Δt*dλv(λv)
    J(λv) = 1 - Δt*2λ/(3η) * hessian(model, λ, λv)
    λv_new = λv_old - R(λv_old) / J(λv_old)
    for _ ∈ 1:20
        res = R(λv_new)
        abs(res) < 1e-8 && return λv_new
        λv_new -= res / J(λv_new)
    end
    λv_new
end

function viscous_evolution(model::GeneralizedMaxwell, Δt::Real, λ::Real, A)
    map((b,λv) -> viscous_evolution(b,Δt,λ,λv), model.branches, A)
end

function viscous_evolution(model::ThermoMechanicalModel, Δt::Real, λ::Real, θ::Real, A)
    viscous_evolution(model.mechano, Δt, λ, A)
end

## Testing

P_autodiff(model,λ) = ForwardDiff.derivative(x -> free_energy_density(model,x), λ)
H_autodiff(model,λ) = ForwardDiff.derivative(x -> P_autodiff(model,x), λ)
carroll = Carroll(6.02e3, 1.22e-3, 2.82e4)
yeoh_1  = Yeoh(3.23e1)
neoh_1  = NeoHooke(1.33e4)
@assert P_autodiff(carroll, 2.87) ≈ stress(carroll, 2.87)
@assert P_autodiff(yeoh_1,  2.87) ≈ stress(yeoh_1,  2.87)
@assert P_autodiff(neoh_1,  2.87) ≈ stress(neoh_1,  2.87)
@assert H_autodiff(yeoh_1, 2.87) ≈ hessian(yeoh_1,  2.87)
@assert H_autodiff(neoh_1, 2.87) ≈ hessian(neoh_1,  2.87)

#endregion
#region Simulations

## Auxiliary functions

function loading_test(x...)
    p = plot()
    loading_test!(x...)
    return p
end

function loading_test(model::ViscoElasticModel, λ_max, v)
    t_max = (λ_max-1) / v
    Δt = t_max / 20
    λ_values = map(t -> min(1+v*t, 2λ_max-1-v*t), 0:Δt:1.6t_max)
    A = ntuple(_ -> 1.0, 5)
    P_values = map(λ_values) do λ
        A = viscous_evolution(model, Δt, λ, A)
        P = stress(model, λ, A)
    end
    label = @sprintf("%3d%%, %.1g/s", 100*(λ_max-1), v)
    plot!(λ_values, P_values, xlabel="Stretch [-]", ylabel="Stress [Pa]", lw=2, label=label, palette=palette_1)
end

function loading_test!(model::ThermoMechanicalModel, λ_max, θ, v)
    t_max = (λ_max-1) / v
    Δt = t_max / 100
    λ_values = map(t -> min(1+v*t, 2λ_max-1-v*t), 0:Δt:1.6t_max)
    A = ntuple(_ -> 1.0, 5)
    P_values = map(λ_values) do λ
        A = viscous_evolution(model, Δt, λ, θ, A)
        P = stress(model, λ, θ, A)
    end
    label = @sprintf("%3d%%, %2dºC, %.1g/s", 100*(λ_max-1), θ-273.15, v)
    plot!(λ_values, P_values./1e6, xlabel="Stretch [-]", ylabel="Stress [MPa]", lw=2, label=label, palette=palette_1)
end

function loading_test!(model::ThermoMechanicalModel, data::Vector{LoadingTest}, λ_max, θ, v)
    record = getfirst(r -> r.λ_max≈λ_max && r.θ≈θ && r.v≈v ,data)
    Δt = record.Δt
    λ_values = record.λ
    A = ntuple(_ -> 1.0, 5)
    P_values = map(λ_values) do λ
        A = viscous_evolution(model, Δt, λ, θ, A)
        P = stress(model, λ, θ, A)
    end
    label = @sprintf("%3d%%, %2dºC, %.1g/s", 100*(λ_max-1), θ-273.15, v)
    plot!(λ_values, P_values./1e6, xlabel="Stretch [-]", ylabel="Stress [MPa]", lw=2, label=label, palette=palette_2)
    scatter!(λ_values, record.σ./1e6, msw=0, label="")
end

function loading_test_cv(model::ThermoMechanicalModel, v)
    function cv_func(λ_max, θ)
        A = ntuple(_ -> 1.0, 5)
        λ_max ≈ 1 && return specific_heat(model, λ_max, θ, A)
        t_max = (λ_max-1) / v
        Δt = t_max / 100
        for λ ∈ map(t -> 1+v*t, 0:Δt:t_max)
            A = viscous_evolution(model, Δt, λ, θ, A)
        end
        return specific_heat(model, λ_max, θ, A)
    end
    λ_values = 1:.1:5
    θ_values = 203.0:5:493
    cv_values = @. cv_func(λ_values', θ_values)
    cv_lim = maximum(abs.(skipmissing(cv_values)))
    cv_values = clamp.(cv_values, -cv_lim, cv_lim)
    diverging_rb = [reverse(palette(:blues,10))...; palette(:OrRd,10)...]
    p = plot(title="Specific heat under isochoric stretch, v=$v/s", xlabel="Stretch [-]", ylabel="θ/θR [-]", framestyle=:grid, rightmargin=8Plots.mm, titlefontsize=10)
    contourf!(λ_values, θ_values./θr, cv_values, color=diverging_rb, clims=(-cv_lim, cv_lim), lw=0)
    plot!([1.02, 3.98, 3.98, 1.02, 1.02], ([-20, -20, 80, 80, -20].+273.15)./θr, color=:black, lw=2, label="")
    return p
end

## Constitutive models

long_term = Carroll(6.02e3, 1.22e-3, 2.82e4)
branch_1 = ViscousBranch(Yeoh(3.23e1), 3.05e2)
branch_2 = ViscousBranch(Yeoh(1.63e-4), 3.95e-4)
branch_3 = ViscousBranch(NeoHooke(1.33e4), 3.43e1)
branch_4 = ViscousBranch(NeoHooke(2.12e3), 5.4e2)
branch_5 = ViscousBranch(NeoHooke(4.5e2), 1.23e5)
maxwell = GeneralizedMaxwell(long_term, (branch_1, branch_2, branch_3, branch_4, branch_5))
g_yeoh(θ) = g1(θ, θr, 2.08e1)
g_neoh(θ) = g2(θ, θr, 1.93e1, 2.21e-1)
laws = (g_yeoh, g_yeoh, g_neoh, g_neoh, g_neoh)
model = ThermoMechanicalModel(maxwell, laws)

## Execute stress-strain plot and cv maps

p = loading_test(model, data, 4.0, θr, 0.1)
loading_test!(model, data, 4.0, θr, 0.05)
loading_test!(model, data, 4.0, θr, 0.03)
display(p)

display(loading_test_cv(model, 0.10))
