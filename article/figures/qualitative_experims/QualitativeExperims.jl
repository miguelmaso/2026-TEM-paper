using Plots

default(lw=3)
default(labelfontsize=12)
default(size=(300,250))
default(label=false)
default(axis=[])

theme = palette(:seaborn_colorblind)
red = theme[4]
blue = theme[1]
green = theme[3]


# Differental scanning calorimetry
dsc(x) = 0.8 + 0.1x + 0.02sin(x)
display(plot(0:.1:5, dsc, xlabel="θ", ylabel="cᵥ", ylims=[0, 1.5], c=blue))


# One-cycle loading-unloading
λ = range(0, 1.2, length=100)
σ1(x) = 1.0 * (1 - exp.(-2.0 * x))                # loading
σ2(x) = 1.5 * (1 - exp.(-1.0 * x)) + 1.0 * (x - 0.6)^2 -0.5  # unloading
plot(λ, σ1, c=red, xlabel="λ", ylabel="σ", ylims=[0,1], xlims=[0,1.3])
display(plot!(λ, σ2, c=red))


# Quasi-static loading

