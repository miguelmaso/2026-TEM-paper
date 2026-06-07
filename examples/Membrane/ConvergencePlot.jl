using JLD2
using Gridap, Gridap.CellData, Gridap.FESpaces
using Plots
using LaTeXStrings

default(
    fontfamily     = "Computer Modern",
    legendfontsize = 12,
    tickfontsize   = 10,
    labelfontsize  = 14,
    titlefontsize  = 12,
    palette        = :seaborn_colorblind,
    linewidth      = 2
)

function load_uh(path)
    @load path uh⁺
    return uh⁺
end

function norm_Linf(uh)
    reffe = ReferenceFE(lagrangian, Float64, 1)
    Ω = get_triangulation(uh)
    V = FESpace(Ω, reffe, conformity=:L2)
    u2 = interpolate_everywhere(uh·uh, V)
    u2_values = [u2.free_values; u2.dirichlet_values]
    sqrt(norm(u2_values, Inf))
end

function error_L2(coarse, fine, order)
    V_ref = get_fe_space(fine)
    Ω = get_triangulation(V_ref)
    dΩ = Measure(Ω, 2*order)
    i_coarse = Interpolable(coarse)
    u = interpolate_everywhere(i_coarse, V_ref)
    err = u - fine
    u_max = norm_Linf(fine)
    sum(∫( err·err )dΩ) / sum(∫( 1.0 )dΩ) / u_max
end

function nslope(order, x, y0)
    y0 .* (x ./ x[1]).^(2*order)
end

res_path = abspath(dirname(@__FILE__), "results/")
fig_path = abspath(dirname(@__FILE__), "../../article/figures/membrane/")

order = 2
divisions = [10,20,30]
max_divisions = 40
spacing = 0.1 ./ divisions

uh_fine = load_uh("$(res_path)/Membrane_uh_$(order)_$(max_divisions).jld2")

e = map(divisions) do n
    uh_coarse = load_uh("$(res_path)/Membrane_uh_$(order)_$(n).jld2")
    err = error_L2(uh_coarse, uh_fine, order)
    return err
end

p = plot(xaxis=:log, yaxis=:log, xlabel="h", ylabel=L"error\ \mathcal{L}_2")
plot!(spacing, e, label="order 2", marker=:x)
plot!(spacing, nslope(order, spacing, e[1]), label="slope 1:4", lw=1, c=:black)

display(p);
savefig(p, abspath(dirname(@__FILE__), "$(fig_path)/convergence.pdf"))
