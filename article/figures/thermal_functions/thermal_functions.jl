using HyperFEM
using Plots

pgfplotsx()  # Comment this line or type `GR()` for a faster (non LaTeX-like) rendering
default(palette   = :seaborn_colorblind)
default(legend    = :topleft)
default(linewidth = 2)

γ = 0.5

law_a = EntropicElasticityLaw(θr=1.0, γ=γ)
law_b = NonlinearMeltingLaw(θr=1.0, θM=2.0, γ=γ)
law_c = NonlinearSofteningLaw(θr=1.0, θt=0.8, γ=4γ, δ=0.5)

fa, dfa, ddfa = derivatives(law_a)
fb, dfb, ddfb = derivatives(law_b)
fc, dfc, ddfc = derivatives(law_c)

labels = ["Entropic elasticity" "Nonlinear melting" "Nonlinear softening"]

p = plot(0:0.01:2, [fa, fb, fc], label=labels, xlabel="θ/θᵣ", ylabel="Ψ/Ψᵣ")
savefig(abspath(@__DIR__, "..", stem(@__FILE__) * ".pdf"))
display(p);
