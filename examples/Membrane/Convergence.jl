using JLD2
using Gridap, Gridap.CellData
using Plots

function load_uh(path)
    @load path uh
    return uh
end

function error_L2(coarse, fine, order)
    V_ref = get_fe_space(fine)
    Ω = get_triangulation(V_ref)
    dΩ = Measure(Ω, 2*order)
    i_coarse = Interpolable(coarse)
    u = interpolate_everywhere(i_coarse, V_ref)
    err = u - fine
    sum(∫( err·err )dΩ)
end

path = abspath(dirname(@__FILE__), "results/Membrane")

uh_1_2 = load_uh("$(path)_uh_1_2.jld2")
uh_1_4 = load_uh("$(path)_uh_1_4.jld2")

error_L2(uh_1_2, uh_1_4_ref, 1)
