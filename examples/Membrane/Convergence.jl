using JLD2
using Gridap, Gridap.CellData
using Plots

function load_uh(path)
    @load path uh
    return uh
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

path = abspath(dirname(@__FILE__), "results/Membrane")


uh_fine = load_uh("$(path)_uh_$(order)_6.jld2")

order = 1
divisions = [2,4]
spacing = 0.1 ./ divisions

e = map(divisions) do n
    uh_coarse = load_uh("$(path)_uh_$(order)_$(n).jld2")
    err = error_L2(uh_coarse, uh_fine, order)
    return err
end

plot(spacing, e, xaxis=:log, yaxis=:log)
