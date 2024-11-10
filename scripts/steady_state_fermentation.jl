using DrWatson
@quickactivate "redox_steady_state_aquifers"

# Here you may include files from the source directory

println(
"""
Currently active project is: $(projectname())

Path of active project: $(projectdir())

Additional script that runs a similar model than the one published in the paper. But in this 
script fermentation is also simulated. The script results confirm the steady state concentrations
predicted in the supporting information of the paper.
"""
)
using DifferentialEquations
using CairoMakie
using Colors
using Statistics
using DataFrames
using LaTeXStrings
using Symbolics

function create_fermentation(v, De, dx, c_in, nmob)
    # Defining the reaction rates model
    function Ca_rate(Fk, B, μₘ, Ya)
        return -μₘ/Ya .* Fk .* B
    end

    function bac_rate(Fk , B, μₘ, k_dec)
        return (μₘ .* Fk .- k_dec) .* B
    end

    function Cf_rate(Fk, B, μₘ, Yd, α, Kb)
        return α.*B./(Kb.+B) - μₘ/Yd .* Fk .* B
    end

    function Cd_rate(Fk1, Fk2, B₁, B₂, μ₁, μ₂, Y₂¹, Y₂)
        return μ₁/Y₂¹ .* Fk1.* B₁ - μ₂/Y₂ .* Fk2 .* B₂
    end

    function fermentation!(du, u, p ,t)
        α = p[1]
        Kb = p[2]
        μ₁ = p[3]
        Kf = p[4]
        Id = p[5]
        Yf = p[6]
        Yd1 = p[7]
        k_dec = p[8]
        μ₂ = p[9]
        Kd = p[10]
        Ka = p[11]
        Ya = p[12]
        Yd2 = p[13]
        k_dec2 = p[14]

        # transport
        c_advec = [c_in;u[:,1:nmob]]
        advec = -v .* diff(c_advec, dims=1) ./ dx
        gradc = diff(u[:,1:nmob], dims=1)./dx
        disp = ([gradc; zeros(1, nmob)]-[zeros(1, nmob); gradc]).* De ./ dx
        
        Fk1 = u[:,2]./(u[:,2].+Kf) .* Id./(Id.+u[:,3])
        Fk2 = u[:,1]./(u[:,1].+Ka) .* u[:,3]./(u[:,3].+Kd)


        du[:,1] .= advec[:,1] .+ disp[:,1] .+ Ca_rate(Fk2, u[:,5], μ₂, Ya)
        du[:,2] .= advec[:,2] .+ disp[:,2] .+ Cf_rate(Fk1, u[:,4], μ₁, Yf, α, Kb)
        du[:,3] .= advec[:,3] .+ disp[:,3] .+ Cd_rate(Fk1, Fk2, u[:,4], u[:,5], μ₁, μ₂, Yd1, Yd2)
        du[:,4] .= bac_rate(Fk1, u[:,4], μ₁, k_dec)
        du[:,5] .= bac_rate(Fk2, u[:,5], μ₂, k_dec2)
    nothing
    end
    return fermentation!, Ca_rate
end

#---- Model running: fermentation-------
# Parameters
nmob = 3
nsub = 5

# Defining the parameters
α = 8e-10
Kb = 5e-5
μ₁ = 1e-5
Kf = 1.3e-5
Id = 1e-6
Yf = 0.3
Yd1 = 0.2
k_dec = 1.15e-6
μ₂ = 5e-5
Kd = 1.e-5
Ka = 5e-4
Ya = 0.3
Yd2 = 0.2
k_dec2 = 1.15e-6

p = [α, Kb, μ₁, Kf, Id, Yf, Yd1, k_dec, μ₂, Kd, Ka, Ya, Yd2, k_dec2]

# investigating steady state concentrations:
Kd/Id < (μ₁/k_dec-1)*(1-k_dec2/μ₂)/(k_dec2/μ₂) ? println("Steady state exists") : println("Steady state does not exist")
μ₁*μ₂/(k_dec*k_dec2)
c2_min = Kd*k_dec2/μ₂/(1-k_dec2/μ₂)
c2_max = Id*(μ₁/k_dec-1)
ca_min = Ka*k_dec2/μ₂/(1-k_dec2/μ₂)
c1_min = Kf*k_dec/μ₁/(1-k_dec/μ₁)
#defining the transport parameters
dx = 0.01 # [m]
L = 12 # [m]
v = 1/86400 # [m/s]
x = 0:dx:L

#defining the transport parameters
αₗ = 0.1           # dispersivity [m] - same for all compounds
Dp = [2.8e-10 5.34e-10 5.34e-10] # pore diffusion coefficient [m^2/s] 
                         # - one value for each mobile compound
D = αₗ * v .+ Dp # dispersion coefficient of all mobile compounds [m^2/s]
# inflow concentration
cₐ = 6e-4
c_in = [cₐ c1_min 0]

#make the ode function
fermentation!, ca_rate_ferm = create_fermentation(v, D, dx, c_in, nmob)

# Initial conditions
u0 = zeros(size(x,1), nsub)
du0 = u0
Bss1 = α*Yf/k_dec - Kb
Bss2 = (α*Yf/Yd1 - Kb*k_dec/Yd1)*Yd2/k_dec2
b₀ = 0.2
rate_ss = Yd2/Ya*(α*Yf/Yd1 - Kb*k_dec/Yd1)
u0[:,4] .= Bss1*b₀
u0[:,5] .= Bss2*b₀
# Time span
tspan = (0.0, 2*365*86400) # [s] 2 years
# creating the problem
jac_sparsity = Symbolics.jacobian_sparsity((du, u) -> fermentation!(du, u, p, 1),
    du0, u0) # add the sparsity pattern to speed up the solution
ferm_rhs! = ODEFunction(fermentation!, jac_prototype=jac_sparsity)
prob = ODEProblem(ferm_rhs!, u0, tspan, p)
# solving the problem
sol = solve(prob, Rosenbrock23(), saveat=86400, reltol = 1e-9, abstol = 1e-12)

num_frames = 100  # Total number of frames
time_values = range(start=log10(1e-10), stop=log10(sol.t[end]), length=num_frames)

fig = Figure(size = (800, 600))
ax = Axis(fig[1, 1], xlabel = "x [m]", ylabel = "reaction rate [mol L⁻¹ s⁻¹]")
ss_line = rate_ss .* ones(size(x,1))
lines!(ax, x, ss_line, color = :darkred, linestyle = :dash, label = "Steady state")
Fk = sol.u[1][:,3]./(sol.u[1][:,3].+Kd) .* sol.u[1][:,1]./(sol.u[1][:,1].+Ka)
lines = lines!(ax, x, -ca_rate_ferm(Fk, sol.u[1][:, 5], μ₂, Ya), color = :blue, label = "Time: $(sol.t[1] / 86400) days")
record(fig, plotsdir("reacrate_animation.mp4"), 1:length(sol.t); framerate = 10) do i
    Fk = sol.u[i][:,3]./(sol.u[i][:,3].+Kd) .* sol.u[i][:,1]./(sol.u[i][:,1].+Ka)
    lines[2] = -ca_rate_ferm(Fk, sol.u[i][:, 5], μ₂, Ya)
    ax.title = "Time: $(sol.t[i]/ 86400) days"
end

function pore_volume(t)
    return v*ϕ*t/L
end
pore_volume.(sol.t)
with_theme(theme_latexfonts()) do
    fig = Figure(size = (800, 600))
    ax = Axis(fig[1, 1], xlabel = "x [m]", xlabelsize = 15,
        xticklabelsize = 18,
        ylabel = L"$C_A$ [mol L⁻¹]", ylabelsize = 15,
        yticklabelsize = 18)
    ax.xticks = 0:2:10
    ax.yticks = 0:1:6
    ylims!(ax, -1e-9, 6.3)
    ss_line = (cₐ .- rate_ss/v .* x).*1e4
    half_sat_line = Ka .* ones(size(x,1)).*1e4
    ca_min_line = ca_min .* ones(size(x,1)).*1e4
    lines!(ax, x, ss_line, color = :darkred, linestyle = :dash, label = "Avection dominated prediction",
        linewidth = 2.5)
    text!(ax, L"C_A = C_A^{in} - r_{ss}/v \times x" ,position=(maximum(x)*0.5, (cₐ .- rate_ss/v .* maximum(x)*0.5).*1e4),
    align = (:left, :bottom), rotation = atan(-rate_ss*1e4/v,1), fontsize = 28)
    lines!(ax, x, half_sat_line, color = :black, linestyle = :dash, label = "Half saturation constant",
        linewidth = 3.5)
    text!(ax, L"K_A" ,position=(maximum(x)*0.3, (Ka+0.1e-4).*1e4), fontsize = 28, font="CMU Serif Bold")
    lines!(ax, x, ca_min_line, color = :darkblue, linestyle = :dash, label = "Minimum concentration",
        linewidth = 3.5)
    text!(ax, L"C_A^{min}" ,position=(maximum(x)*0.3, (ca_min+0.1e-4).*1e4), fontsize = 28)
    lines = lines!(ax, x, sol.u[1][:,1].*1e4, color = :blue, label = "Time: $(sol.t[1] / 86400) days",
        linewidth = 2.9)
    Label(fig[1, 1, Top()], halign = :left, L"\times 10^{-4}")
    record(fig, plotsdir("ca_animation_ferm.mp4"), 1:length(sol.t); framerate = 30) do i
        lines[2] = sol.u[i][:, 1].*1e4
        ax.title = "Pore Volume: $(round(pore_volume.(sol.t[i]);digits=2))"
    end
    # save last frame as svg
    save(plotsdir("ca_animation_ferm.svg"), fig)
    save(plotsdir("ca_animation_ferm.png"), fig, px_per_unit = 1200/96)
end

# add a color range
colormap = cgrad(:deep, norms)
ashgray = "#B2BEB5"
glaucous = "#6082B6"

with_theme(theme_latexfonts()) do
    fig = Figure(size = (504, 500))
    ax = Axis(fig[1, 1:4], xlabel = "x [m]", xlabelsize = 15,
        xticklabelsize = 14,
        ylabel = L"$r_{C_A}$ [mol L⁻¹ s⁻¹]", ylabelsize = 15,
        yticklabelsize = 14)
    ax.xticks = 0:2:12
    ylims!(ax, -1e-9, 6.5)
    ss_line = rate_ss .* ones(size(x,1)).*1e10
    bss_line = Bss1 .* ones(size(x,1)).*1e4
    Label(fig[1, 1:4, Top()], halign = :left, L"\times 10^{-10}")
    Label(fig[1, 1:4, Top()], halign = :right, "a.", fontsize = 20,
    font = :bold)
    lines!(ax, x, ss_line, color = :darkred, linestyle = :dash, label = "steady-state reaction rate",
        linewidth = 2.5)
    Fk = sol.u[1][:,3]./(sol.u[1][:,3].+Kd) .* sol.u[1][:,1]./(sol.u[1][:,1].+Ka)
    lines = lines!(ax, x, -1e10.*ca_rate_ferm(Fk, sol.u[1][:, 5], μ₂, Ya), color = (scale[1],0.6), label = "Time: $(sol.t[1] / 86400) days",
        linewidth = 2.9)
    for i in 1:length(sol.t)
        Fk = sol.u[i][:,3]./(sol.u[i][:,3].+Kd) .* sol.u[i][:,1]./(sol.u[i][:,1].+Ka)
        lines = lines!(ax, x, -1e10.*ca_rate_ferm(Fk, sol.u[i][:, 5], μ₂, Ya), color = (scale[i],0.6), label = "Time: $(sol.t[i] / 86400) days",
        linewidth = 2.9)
    end
    ax2 = Axis(fig[2,1:4], xlabel = "x [m]", xlabelsize = 15,
        xticklabelsize = 14,
        ylabel = "B [mol L⁻¹]", ylabelsize = 15,
        yticklabelsize = 14)
    ylims!(ax2, -1e-9, 2)
    lines!(ax2, x, bss_line, color = :darkred, linestyle = :dash, label = "Steady-state biomass concentration",
        linewidth = 2.5)
    l = lines = lines!(ax2, x, sol.u[1][:,4].*1e4, color = (scale[1],0.6), label = "Time: $(sol.t[1] / 86400) days",
        linewidth = 2.9)
    l2 = lines!(ax2, x, sol.u[1][:,5].*1e4, color = (scale[1],0.6), label = "Time: $(sol.t[1] / 86400) days",
        linewidth = 2.9)
    for i in 1:length(sol.t)
        lines = lines!(ax2, x, sol.u[i][:,4].*1e4, color = (scale[i],0.6), label = "Time: $(sol.t[i] / 86400) days",
        linewidth = 2.9)
        lines2 = lines!(ax2, x, sol.u[i][:,5].*1e4, color = (scale[i],0.6), label = "Time: $(sol.t[i] / 86400) days",
        linewidth = 2.9)
    end

    Label(fig[2, 1:4, Top()], halign = :left, L"\times 10^{-4}")
    # add the scale bar
    Label(fig[2, 1:4, Top()], halign = :right, "b.", fontsize = 20,
    font = :bold)
    Colorbar(fig[1:2, 5], limits = (minimum(sol.t), maximum(sol.t))./86400, colormap=scale, label = "Time [days]", labelsize = 15, ticklabelsize = 14)
    fig
    save(plotsdir("reactBss_ferm.svg"), fig)
    save(plotsdir("reactBss_ferm.png"), fig, px_per_unit = 1200/96)
end