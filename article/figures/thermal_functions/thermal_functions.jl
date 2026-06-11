using HyperFEM
using Plots
using LaTeXStrings

pgfplotsx()  # Comment this line or type `gr()` for a faster (non LaTeX-like) rendering
default(fontfamily     = "Computer Modern")
default(palette        = :seaborn_colorblind)
default(legend         = :top)
default(linewidth      = 3)
default(legendfontsize = 12)
default(tickfontsize   = 12)
default(labelfontsize  = 12)

γ = 0.5

law_a = EntropicElasticityLaw(θr=1.0, γ=γ)
law_b = NonlinearMeltingLaw(θr=1.0, θM=2.0, γ=γ)
law_c = NonlinearSofteningLaw(θr=1.0, θT=0.8, γ=2γ, δ=0.5)
law_d = ConstantCvLaw(θr=1.0)
laws = [law_a, law_b, law_c, law_d]
funcs = map(law -> law()[1], laws)

labels = ["Entropic elasticity" "Nonlinear melting" "Nonlinear softening" "Constant cᵥ"]

p = plot(0:0.01:2, funcs, label=labels, xlabel=L"\theta / \theta_R", ylabel=L"\Psi / \Psi_R")
# savefig(abspath(dirname(@__FILE__), "..", stem(@__FILE__) * ".pdf"))
display(p);
