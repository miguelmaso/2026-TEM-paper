using HyperFEM
using Gridap
using CairoMakie

include(joinpath(dirname(@__FILE__), "../Membrane/Membrane.jl"))



"""Return the linear index of a N-dimensional tensor"""
_flat_idx(i::Int, j::Int, N::Int) = i + N*(j-1)
_flat_idx(i::Int, j::Int, k::Int, l::Int, N::Int) = _flat_idx(_flat_idx(i,j,N), _flat_idx(k,l,N), N*N)

"""Return the cartesian indices of an N-dimensional second-order tensor"""
_full_idx2(α::Int, N::Int) = ((α-1)%N+1 ,(α-1)÷N+1)

"""Return the cartesian indices of an N-dimensional fourth-order tensor"""
_full_idx4(α::Int, β::Int, N::Int) = (_full_idx2(α,N)..., _full_idx2(β,N)...)
_full_idx4(α::Int, N::Int) = _full_idx4(_full_idx2(α,N*N)...,N)




"""
    contraction_IJKL_JL(A::TensorValue{D*D}, H::TensorValue{D})::TensorValue{D*D}

Performs a tensor contraction between a fourth-order tensor (represented as a `D² × D²` matrix in flattened index notation)
and a second-order tensor (of size `D × D`).
The operation follows the **index contraction pattern**, where addition is performed for repeated indices.
"""
@generated function contraction_IJKL_JL(H::TensorValue{D²}, A::TensorValue{D}) where {D, D²}
  @assert D*D == D² "Fourth- and Second-order tensors size mismatch"
  str = ""
  for i in 1:D
    for k in 1:D
      for j in 1:D
        for l in 1:D
          a = _flat_idx(i,j,k,l,D)
          str *= "+H[$a]*A[$j,$l]"
        end
      end
      str *= ","
    end
  end
  Meta.parse("TensorValue{D}($str)")
end

(⊙ᵢⱼₖₗʲˡ) = contraction_IJKL_JL


function sylvester_num(A::TensorValue{3})
  minor_1 = A[1]
  minor_2 = det(A[1:2,1:2])
  minor_3 = det(A)
  min(minor_1, minor_2, minor_3)
end


model = build_model(θr=293.15)
∂∂Ψ∂F∂F = model()[5]
P = model()[2]

F1 = TensorValue(2.5, 0.0, 0.0, 0.0, (2.5)^(-1/2), 0.0, 0.0, 0.0, (2.5)^(-1/2))
F0 = TensorValue(2.3, 0.0, 0.0, 0.0, (2.3)^(-1/2), 0.0, 0.0, 0.0, (2.3)^(-1/2))
update_time_step!(model, 1.0)

E0 = VectorValue(0.0, 5000/0.0005, 0.0)
θ1 = 293.3

Ai = VectorValue(F0..., 0.0)
A = (Ai, Ai, Ai)

function evaluate_positiveness_acoustic_tensor(α::Float64, β::Float64)
  n = VectorValue(cos(α)*cos(β), sin(β)*cos(α), sin(β))
  a = ∂∂Ψ∂F∂F(F1, E0, θ1, F0, A...) ⊙ᵢⱼₖₗʲˡ (n⊗n)
  sylvester_num(a)
end


n_points = 20
x = [range(0,2π, length=n_points)...]
y = [range(0,π, length=n_points)...]
Z = evaluate_positiveness_acoustic_tensor.(x, y')

fig = Figure()
# ax = Axis3(fig[1, 1], zgridvisible = false, xgridvisible = true, ygridvisible = true) # Quita rejillas

ax = Axis3(fig[1, 1],
      xlabel = "α",
      ylabel = "β",
      
      # 2) Cambiamos la escala de la caja usando 'aspect'.
      # Como α (2π ≈ 6.28) es el doble que β (π ≈ 3.14), usamos (2, 1, X).
      # Modifica el tercer número (0.6) para estirar o achatar el eje Z a tu gusto.
      aspect = (2, 1, 0.1), 
      
      # 3) Mostramos SOLO el plano horizontal (suelo) y sus rejillas
      # zpanelvisible = true,   # Activa el suelo (plano XY)
      # xpanelvisible = false,  # Oculta la pared YZ
      # ypanelvisible = false,  # Oculta la pared XZ
      xgridvisible = false,    # Rejilla del suelo en X
      ygridvisible = false,    # Rejilla del suelo en Y
      zgridvisible = false,   # Quita líneas horizontales elevadas
      
      # 4) Hacemos desaparecer por completo el eje Z y líneas verticales
      zticksvisible = false,
      zticklabelsvisible = false,
      # zspinevisible = false,
      zlabelvisible = false
)

CairoMakie.surface!(ax, x, y, Z, color = Z, colorrange = (minimum(Z), maximum(Z)), colormap = :roma)

Colorbar(fig[1, 2], limits = (minimum(Z), maximum(Z)), colormap = :roma)
display(fig)



# contourf(x, y, Z; 
#   color = :roma,
#   xlabel = "α", 
#   ylabel = "β",
#   lw=0, lc=:transparent,
#   levels=20)