using Plots
using HyperFEM
using Gridap

default(linewidth=3)
default(palette=:seaborn_colorblind)

μe1 = 4.6e2   # [Pa]
μe2 = 3.8e4   # [Pa]
α1  = 2.0     # [-]
α2  = 1.3     # [-]

nh = NeoHookean3D(μ=μe1+μe2, λ=0.0)
mr = NonlinearMooneyRivlin3D(μ1=μe1, μ2=μe2, α1=α1, α2=α2, λ=0.0)

F_1(λ) = TensorValue(λ, 0, 0, 0, λ^(-1/2), 0, 0, 0, λ^(-1/2))
F_2(λ) = TensorValue(λ, 0, 0, 0, λ, 0, 0, 0, λ^(-2))

Ψnh, dΨnh_dF, ddΨnh_dFF = nh()
Ψmr, dΨmr_dF, ddΨmr_dFF = mr()

Pnh_1(λ) = getindex.(dΨnh_dF(F_1(λ)), 1)
Pmr_1(λ) = getindex.(dΨmr_dF(F_1(λ)), 1)

Pnh_2(λ) = getindex.(dΨnh_dF(F_2(λ)), 9)
Pmr_2(λ) = getindex.(dΨmr_dF(F_2(λ)), 9)

λ_values_1 = exp10.(range(-1, 1, length=50))
λ_values_2 = range(1, 3, length=50)

params = (label=["Neo-Hooke" "Mooney-Rivlin"], xlabel="Stretch, λ [-]", ylabel="Stress [Pa]")
p1 = plot(λ_values_1, [Pnh_1, Pmr_1]; params..., title="Uniaxial stretch", xaxis=:log)
p2 = plot(λ_values_2, [Pnh_2, Pmr_2]; params..., title="Equibiaxial stretch")

display(p1);
display(p2);
