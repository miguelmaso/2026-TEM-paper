using JLD2
using Plots, Plots.Measures
using Peaks
using LaTeXStrings

gr()

default(
    fontfamily     = "Computer Modern",
    legendfontsize = 12,
    tickfontsize   = 10,
    labelfontsize  = 14,
    titlefontsize  = 12,
    palette        = :seaborn_colorblind,
    linewidth      = 2
)

m = jldopen("results_100cycles/Membrane_metrics_3.0.jld2") do f
  f["metrics"]
end

Δt = 0.02
prestretch = 3.0

# Metrics visualization and check

η_ref = m.ηtot[1]
p1 = Plots.plot(m.time, m.ηtot, labels="Entropy", style=:solid, lcolor=:black, width=2, ylim=[1-5.1e-3, 1+5.1e-3]*η_ref, yticks=[1-5e-3, 1, 1+5e-3]*η_ref, margin=8mm, xlabel="Time [s]", ylabel="Entropy [J/K]")
p1 = Plots.plot!(p1, m.time, NaN.*m.time, labels="Temperature", style=:dash, lcolor=:gray, width=2)
p1 = Plots.plot!(twinx(p1), m.time, m.θavg, labels="Temperature", style=:dash, lcolor=:gray, width=2, xticks=false, legend=false, ylabel="Temperature [ºK]")

Ψint = m.Ψmec + m.Ψele + m.Ψthe
Ψtot = Ψint - m.Ψdir
p2 = Plots.plot(m.time, [Ψint m.Ψdir m.Dvis], labels=["̇Ψu+Ψφ+Ψθ" "Ψφ,Dir" "Dvis"], style=[:solid :dash :dashdot], lcolor=[:black :black :gray], width=2, margin=8mm, xlabel="Time [s]", ylabel="Power [W]")

p4 = Plots.plot(p1, p2, layout=@layout([a b]), size=(1500, 500))
display(p4);


trapz(a::AbstractArray) = sum(a) -0.5(a[1] + a[end])
moving_average(a::AbstractArray, n) = [sum(@view a[max(1,i-n+1):max(n,i)]) / n for i in eachindex(a)]

Dvis_θ = m.Dvis ./ m.θavg
Dvis_int = trapz(Dvis_θ) * Δt
@show m.ηtot[end] - m.ηtot[1]
@show m.ηtot[end] - m.ηtot[1] - Dvis_int

@show trapz(Dvis_θ ./ m.cv)
@show trapz(m.∂Pθ_F ./ m.cv)
@show trapz(m.∂Dθ_E ./ m.cv)


p5 = Plots.plot(m.λ, m.V ./1000, color=1, alpha=.5, lw=1, label=nothing, xlabel="Stretch [-]", ylabel="Voltage [kV]", size=(800,1000), margins=8mm)

p6 = Plots.plot(m.time, m.λ, lcolor=1, alpha=.5, lw=1, label=nothing, xlabel="Time [s]", ylabel="Stretch [-]", size=(1200,400), margins=12mm)
Plots.plot!(p6, m.time, moving_average(m.λ,50), lcolor=1, lw=2, label=nothing)
p6_twin = twinx(p6)
Plots.plot!(p6_twin, m.time, m.Dvis .* 1e5, lcolor=2, alpha=.5, lw=1, label=nothing, ylabel="Dissipation [×10⁻⁵ W]", ytickfontcolor=:red, yguidefontcolor=:red)
Plots.plot!(p6_twin, m.time, moving_average(m.Dvis,50) * 1e5, lcolor=2, lw=2, label=nothing)

display(p6);

p7 = Plots.plot(p5, p6, layout=@layout([a{0.33w} b]), size=(1500, 500))

p8 = Plots.plot(m.time, m.θavg .- 273.15, lw=1, lcolor=:black, label=nothing, xlabel="Time [s]", ylabel="Temperature [ºC]", size=(1200,400), margins=12mm)


i_max, λ_max = findmaxima(m.λ)
θavg = m.θavg .- 273.15
p11 = plot(m.time, θavg,            alpha=.8, lw=1, lcolor=1, margins=3mm, label="", xformatter=_->"", ylabel=L"\theta\ [ºC]")
plot!(     m.time, moving_average(θavg,50),   lw=3, lcolor=1, margins=3mm, label="")
p12 = plot(m.time, m.λ,             alpha=.8, lw=1, lcolor=2, margins=3mm, label="", xformatter=_->"", ylabel=L"\lambda\ [-]")
plot!(     m.time[i_max], λ_max,              lw=3, lcolor=2, margins=3mm, label="")
p13 = plot(m.time, m.Dvis.*1e5,          alpha=.8, lw=1, lcolor=4, margins=3mm, label="", xlabel="Time [s]", ylabel=L"\mathcal{D}_{\rm{int}}\ [W\times10^{-5}]")
plot!(     m.time, moving_average(m.Dvis,50), lw=3, lcolor=4, margins=3mm, label="")
plot!([9,10,10,9,9],[0,0,4e-5,4e-5,0], lw=1, lcolor=:black, label="")
plot!(m.time[9000:end], m.Dvis[9000:end], alpha=.8, lw=1, lcolor=4, label="", xformatter=_->"", yformatter=_->"", inset=(1, bbox(0.6,0.0,0.3,0.6)), subplot=2, framestyle=:box)
plot(p11, p12, p13, layout=(3,1), link=:x, size=(1000,600))
