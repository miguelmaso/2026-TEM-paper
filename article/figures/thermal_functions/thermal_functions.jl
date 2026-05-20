using HyperFEM
using Plots
using LaTeXStrings

pgfplotsx()  # Comment this line or type `GR()` for a faster (non LaTeX-like) rendering
default(palette        = :seaborn_colorblind)
default(legend         = :topleft)
default(linewidth      = 3)
default(legendfontsize = 12)
default(tickfontsize   = 12)
default(labelfontsize  = 12)

γ = 0.5

law_a = EntropicElasticityLaw(θr=1.0, γ=γ)
law_b = NonlinearMeltingLaw(θr=1.0, θM=2.0, γ=γ)
law_c = NonlinearSofteningLaw(θr=1.0, θT=0.8, γ=4γ, δ=0.5)
law_d = ConstantCvLaw(θr=1.0)

fa, dfa, ddfa = law_a()
fb, dfb, ddfb = law_b()
fc, dfc, ddfc = law_c()
fd, dfd, ddfd = law_d()

labels = ["Entropic elasticity" "Nonlinear melting" "Nonlinear softening" "Constant cᵥ"]

p = plot(0:0.01:2, [fa, fb, fc, fd], label=labels, xlabel=L"\theta / \theta_R", ylabel=L"\Psi / \Psi_R")
# savefig(abspath(dirname(@__FILE__), "..", stem(@__FILE__) * ".pdf"))
display(p);
