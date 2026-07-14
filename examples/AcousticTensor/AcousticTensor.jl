using Revise
using HyperFEM
using Gridap
using CairoMakie
using Printf
using LaTeXStrings

include(joinpath(dirname(@__FILE__), "../Membrane/Membrane.jl"))


function scientific_notation(v)
  str = @sprintf("%.1e", v) 

  parts = split(str, 'e')
  coeff = parts[1]
  exponent = string(parse(Int, parts[2]))  # "+08"  --> "8"

  if exponent == "0"
    return latexstring(coeff)
  else
    return latexstring(coeff * " \\times 10^{" * exponent * "}")
  end
end


function surface_plot(f)
  n_points = 100
  x = [range(0,2π, length=2n_points)...]
  y = [range(0,π, length=n_points)...]
  Z = f.(x, y')

  xticks = ([0, π/2, π, 3π/2, 2π], ["0", "π/2", "π", "3π/2", "2π"])
  yticks = ([0, π/2, π],           ["0", "π/2", "π"])
  cmap = cgrad(:jet, 8, categorical = true)

  fig = Figure()

  ax = Axis3(
    fig[1, 1],
    xlabel = "α",
    ylabel = "β",
    aspect = (2, 1, 0.1),
    xticks = xticks,
    yticks = yticks,

    xypanelvisible = true,
    xzpanelvisible = false,
    yzpanelvisible = false,
    xypanelcolor = :lightgray,

    xgridvisible = false,
    ygridvisible = false,
    zgridvisible = false,

    zticksvisible = false,
    zticklabelsvisible = false,

    xspinecolor_3 = :transparent,
    yspinecolor_3 = :transparent,
    zspinesvisible = false,
  )
  surf = surface!(ax, x, y, Z, color = Z, colormap = cmap)
  limits!(ax, 0, 2π, 0, π, 0, maximum(Z))

  Colorbar(
    fig[1, 2], surf,
    spinewidth = 0,
    width = 12,
    height = Relative(0.4),
    halign=:right,
    valign=:bottom
  )
  display(fig)
end


function polar_plot(f)
  n_points = 200
  x = [range(0,2π, length=2n_points)...]
  y = [range(0,π, length=n_points)...]
  C = f.(x, y')

  X = C .* (sin.(x) * sin.(y'))
  Y = C .* (cos.(x) * sin.(y'))
  Z = C .* (one.(x) * cos.(y'))

  cmap = cgrad(:jet, 8, categorical = true)

  fig = Figure(size = (900, 700), fontsize = 16)

  ax = Axis3(fig[1, 1], 
    aspect = :data,
    xgridvisible = false,
    ygridvisible = false,
    zgridvisible = false,
    xspinesvisible = false,
    yspinesvisible = false,
    zspinesvisible = false,
    xticksvisible = false,
    yticksvisible = false,
    zticksvisible = false,
    xticklabelsvisible = false,
    yticklabelsvisible = false,
    zticklabelsvisible = false,
    xlabel = "",
    ylabel = "",
    zlabel = "",
  )
  surf = surface!(ax, X, Y, Z, color = C, colormap = cmap)
  # limits!(ax, minimum(X), maximum(X), minimum(Y), maximum(Y), minimum(Z), maximum(Z))

  Colorbar(
    fig[1, 1], surf,
    spinewidth = 0,
    width = 12,
    height = Relative(0.4),
    halign = :right,
    valign = :bottom,
    tellwidth = false,
    tellheight = false,
    alignmode = Outside(-150, 100, 50, 0),
  )

  L = 0.25 * maximum(C) 
  ox = maximum(X) - L/1.5
  oy = minimum(Y) - L/1.5
  oz = maximum(Z) - L/1.5

  arrows3d!(ax,
    [ox, ox, ox], [oy, oy, oy], [oz, oz, oz],
    [-L,  0,  0], [ 0, -L,  0], [ 0,  0,  L],
    color = :black,
    shaftradius = 0.01,
    tipradius = 0.05,
    tiplength = 0.1
  )

  text!(ax,
    [ox + 1.1L, ox, ox], 
    [oy, oy + 1.1L, oy], 
    [oz, oz, oz + 1.3L],
    text = ["X₁", "X₂", "X₃"],
    align = (:center, :center),
    color = :black,
    fontsize = 16
  )

  display(fig)
end


## Gauss point model

model = build_model(θr=293.15)

F1 = TensorValue(2.5, 0.0, 0.0, 0.0, (2.5)^(-1/2), 0.0, 0.0, 0.0, (2.5)^(-1/2))
F0 = TensorValue(2.3, 0.0, 0.0, 0.0, (2.3)^(-1/2), 0.0, 0.0, 0.0, (2.3)^(-1/2))
update_time_step!(model, 1.0)

E0 = VectorValue(0.0, 5000/0.0005, 0.0)
θ1 = 293.3

Ai = VectorValue(F0..., 0.0)
A = (Ai, Ai, Ai)

P    = model()[2]
H_FF = model()[5]
H_EF = model()[8]
H_θF = model()[9]

Hi_FF = H_FF(F1, E0, θ1, F0, A...)
Hi_EF = H_FF(F1, E0, θ1, F0, A...)
Hi_θF = H_FF(F1, E0, θ1, F0, A...)


## Positive definite

function sylvester_num(A::TensorValue{3})
  minor_1 = A[1]
  minor_2 = det(A[1:2,1:2])
  minor_3 = det(A)
  min(minor_1, minor_2, minor_3)
end

function acoustic_tensor_positiveness(α::Float64, β::Float64)
  n = VectorValue(cos(α)*sin(β), sin(α)*sin(β), cos(β))
  a = Hi_FF ⊗₁₂₃₄²⁴ (n⊗n)
  sylvester_num(a)
end

surface_plot(acoustic_tensor_positiveness)


## Polar representations

function H_FF_bulk(α, β)
  n = VectorValue(sin(α)*sin(β), cos(α)*sin(β), cos(β))
  N = n ⊗ n
  N ⊙ (Hi_FF ⊙ N)
end

function H_FF_shear(α, β)
  n = VectorValue(sin(α)*sin(β), cos(α)*sin(β), cos(β))
  error("Not implemented")
end

function H_EF_elec(α, β)
  n = VectorValue(sin(α)*sin(β), cos(α)*sin(β), cos(β))
  N = n ⊗ n
  n ⊙ (Hi_EF ⊙ N)
end

function H_θF_therm(α, β)
  n = VectorValue(sin(α)*sin(β), cos(α)*sin(β), cos(β))
  N = n ⊗ n
  Hi_θF ⊙ N
end


surface_plot(H_FF_bulk)

polar_plot(H_FF_bulk)

