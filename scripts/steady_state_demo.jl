using DrWatson
@quickactivate "redox_steady_state_aquifers"

# Here you may include files from the source directory

println(
"""
Currently active project is: $(projectname())

Path of active project: $(projectdir())

Demonstration of the Simulation of a flow-path depicting a baceria mediated redox reaction and
with electron donor supplied from the solid-phase by hydrolysis.
"""
)
using DifferentialEquations
using CairoMakie
using Colors
using Statistics
using DataFrames
using LaTeXStrings
using CSV
using Symbolics


function create_fixed_decay(v, De, dx, c_in, nmob)
    # Defining the reaction rates model
    function monod_rate(Ca, Cd , B, μₘ, Ka, Kd, Ya,)
        return -μₘ/Ya .* Ca ./ (Ka .+ Ca) .*
         Cd ./ (Kd .+ Cd) .* B
    end

    function bac_rate(Ca, Cd , B, μₘ, Ka, Kd, k_dec)
        return (μₘ .* Ca ./ (Ka .+ Ca) .*
        Cd ./ (Kd .+ Cd) .- k_dec) .* B
    end

    function doc_rate(Ca, Cd, B, Cs, μₘ, Ka, Kd, Yd, α, Kb,Ks, k_dec, η)
        return α.*B./(Kb.+B) .* Cs./(Cs.+Ks) - μₘ/Yd .* Ca ./ (Ka .+ Ca) .*
        Cd ./ (Kd .+ Cd) .* B + η*k_dec .* B
    end

    function org_carbon_rate(B,Cs, α, Kb, Ks)
        return -α.*B./(Kb.+B).* Cs./(Cs.+Ks)
    end

    function fixed_decay!(du, u, p ,t)
        α = p[1]
        Kb = p[2]
        Ks = p[9]
        μₘ = p[3]
        Ka = p[4]
        Kd = p[5]
        Ya = p[6]
        Yd = p[7]
        k_dec = p[8]
        η = p[10]


        # transport
        c_advec = [c_in;u[:,1:nmob]]
        advec = -v .* diff(c_advec, dims=1) ./ dx
        gradc = diff(u[:,1:nmob], dims=1)./dx
        disp = ([gradc; zeros(1, nmob)]-[zeros(1, nmob); gradc]).* De ./ dx

        du[:,1] .= advec[:,1] .+ disp[:,1] .+ monod_rate(u[:,1], u[:,2], u[:,3], μₘ, Ka, Kd, Ya)
        du[:,2] .= advec[:,2] .+ disp[:,2] .+ doc_rate(u[:,1], u[:,2], u[:,3], u[:,4], μₘ, Ka, Kd, Yd, α, Kb, Ks, k_dec, η)
        du[:,3] .= bac_rate(u[:,1], u[:,2], u[:,3], μₘ, Ka, Kd, k_dec)
        du[:,4] .= org_carbon_rate(u[:,3], u[:,4], α, Kb, Ks)
    nothing
    end
    return fixed_decay!, monod_rate
end



## ---- normal parameters model ---------
# Parameters
nmob = 2
nsub = 4

# Defining the parameters
α = 8e-10 #[mol L⁻¹ s⁻¹]
Kb = 5e-5 #[mol L⁻¹] (equivalent water concentration)

μₘ = 1e-5 #[s⁻¹]
Ka = 5e-4 #[mol L⁻¹]
Yd = 0.3 #[mol C_biomass/mol CD_substrate]
γa = 1/25 # stoichiometric coefficient of e-acceptor in the anabolic reaction
γc = 4/5 # stoichiometric coefficient of e-acceptor in the catabolic reaction
Ya = 1/(γa+(1/Yd-1)*γc) # [mol C_biomass/mol CA_substrate]
k_dec = 1.15e-6 #[s⁻¹]
Kd = 1e-6 #[mol L⁻¹]
η = 0.1 # [mol CD_substrate/mol C_biomass]
Ks = 0.22*(α/(k_dec*(1/Yd-η)) - Kb)




#defining the transport parameters
dx = 0.01 # [m]
L = 10 # [m]
v = 0.5/86400 # [m/s]
x = 0:dx:L

#dispersion parameters
αₗ = 0.1           # dispersivity [m] - same for all compounds
Dp = [2.8e-10 5.34e-10] # pore diffusion coefficient [m^2/s] 
                         # - one value for each mobile compound
D = αₗ * v .+ Dp # dispersion coefficient of all mobile compounds [m^2/s]

# inflow concentration
cₐ = 6e-4
cd_min = Kd*k_dec/μₘ/(1-k_dec/μₘ)
ca_min = Ka*k_dec/μₘ/(1-k_dec/μₘ)
c_in = [cₐ cd_min]




fixed_decay!, ca_rate_fd= create_fixed_decay(v, D, dx, c_in, nmob)


#---- Model running: fixed decay+ maintenance-------
# Initial conditions
u0 = zeros(size(x,1), nsub)
du0 = u0
Bss = α/(k_dec*(1/Yd-η)) - Kb
b₀ = 0.1
rate_ss = α/(Ya*(1/Yd-η))-Kb*k_dec/Ya
u0[:,3] .= Bss*b₀
Cs₀ = 1 # mol/kg_sed
ρₛ = 2.65 # [g/cm³]/[kg/L]
ϕ = 0.3
cf = (1-ϕ)*ρₛ/ϕ # [conversion from molC/kg_sed to molC/L_water]
Cs₀ = cf*Cs₀
u0[:,4] .= Cs₀

# parameter vector and table:
p = [α, Kb, μₘ, Ka, Kd, Ya, Yd, k_dec, Ks, η]
ref = ["a", "b", "c", "d", "d", "c", "c", "e", missing, missing,]
description = ["Maximum hydrolysis rate", "Half-saturation constant of biomass",
"Maximum growth rate", "Half-saturation constant of electron acceptor", "Half-saturation constant of electron donor",
"Yield of biomass on electron acceptor", "Yield of biomass on electron donor", "Decay rate constant",
"Half-saturation constant of sediment organic carbon", "proportion of liable biomass decayed"]
symbol = [L"\alpha", L"K_B", L"\mu_{max}", L"K_A", L"K_D", L"Y_A", L"Y_D", L"k_{dec}", L"K_S", L"\eta"]
df = DataFrame(Dict("Symbols"=>symbol, "Values"=> p , "Description" => description, "References"=> ref))
CSV.write(datadir("sims","parameters.csv"), df)
select!(df, [:Symbols, :Values, :Description, :References])
# steady-state Bss in cells/g_sed
smass_bio = 24.6 # [g/cmolbio]
cells_pm = Bss * smass_bio * ϕ/ (100*1e-15) # [cells/L]
cells_g = cells_pm/(1-ϕ)/(ρₛ*1000) # [cells/g_sed]
# Time span
tspan = (0.0, 2*365*86400) # [s] 2 years
# creating the problem
jac_sparsity = Symbolics.jacobian_sparsity((du, u) -> fixed_decay!(du, u, p, 1),
    du0, u0) # add the sparsity pattern to speed up the solution
fixed_rhs! = ODEFunction(fixed_decay!, jac_prototype=jac_sparsity)
prob = ODEProblem(fixed_rhs!, u0, tspan, p)
# solving the problem
sol = solve(prob, Rosenbrock23(), saveat=86400, reltol = 1e-9, abstol = 1e-12)
function pore_volume(t)
    return v*ϕ*t/L
end
pore_volume.(sol.t)
num_frames = 5  # Total number of frames
time_values = [0., 10, 60, 100, 365, 10*365, 100*365] .* 86400
color_values = log.(time_values .+ 1)
norms = color_values ./ maximum(color_values)
# add a color range
colormap = cgrad(:deep, norms)
ashgray = "#B2BEB5"
glaucous = "#6082B6"

with_theme(theme_latexfonts()) do
    fig = Figure(backgroundcolor = RGBf(0.98, 0.98, 0.98), size = (800, 600))
    ax = Axis(fig[1, 1], xlabel = "x [m]", xlabelsize = 30,
        xticklabelsize = 18,
        ylabel = L"$C_A$ [mol L⁻¹]", ylabelsize = 30,
        yticklabelsize = 18)
    ax.xticks = 0:2:10
    ax.yticks = 0:1e-4:6e-4
    ylims!(ax, -1e-9, 6.3e-4)
    ss_line = cₐ .- rate_ss/v .* x
    half_sat_line = Ka .* ones(size(x,1))
    ca_min_line = ca_min .* ones(size(x,1))
    lines!(ax, x, ss_line, color = :darkred, linestyle = :dash, label = "Advection dominated prediction",
        linewidth = 2.5)
    text!(ax, L"C_A = C_A^{in} - r_{ss}/v \times x" ,position=(maximum(x)*0.5, cₐ .- rate_ss/v .* maximum(x)*0.5),
    align = (:left, :bottom), rotation = atan(-rate_ss/v,1)*180/π, fontsize = 28)
    lines!(ax, x, half_sat_line, color = :black, linestyle = :dash, label = "Half saturation constant",
        linewidth = 3.5)
    text!(ax, L"K_A" ,position=(maximum(x)*0.3, Ka+0.1e-4), fontsize = 28, font="CMU Serif Bold")
    lines!(ax, x, ca_min_line, color = :darkblue, linestyle = :dash, label = "Minimum concentration",
        linewidth = 3.5)
    text!(ax, L"C_A^{min}" ,position=(maximum(x)*0.3, ca_min+0.1e-4), fontsize = 28)
    lines = lines!(ax, x, sol.u[1][:,1], color = (glaucous, 0.6), label = "PV: $(pore_volume(sol.t[1])) days",
        linewidth = 2.9)
    record(fig, plotsdir("ca_animation.gif"), 1:length(sol.t); framerate = 30) do i
        lines[2] = sol.u[i][:, 1]
        ax.title = "PV: $(pore_volume(sol.t[i])) days"
    end
end

# make a color scale for the time variable
color_values = (sol.t)
scale = cgrad(:deep, color_values ./ maximum(color_values))
# Plot the reaction rate as function of time and the outflowing concentration of the elements
with_theme(theme_latexfonts()) do
    fig = Figure(size = (504, 500))
    ax = Axis(fig[1, 1:4], xlabel = "x [m]", xlabelsize = 15,
        xticklabelsize = 14,
        ylabel = L"$r_{C_A}$ [mol L⁻¹ s⁻¹]", ylabelsize = 15,
        yticklabelsize = 14)
    ax.xticks = 0:2:10
    ylims!(ax, -1e-9, 5)
    ss_line = rate_ss .* ones(size(x,1)).*1e10
    bss_line = Bss .* ones(size(x,1)).*1e4
    Label(fig[1, 1:4, Top()], halign = :left, L"\times 10^{-10}")
    Label(fig[1, 1:4, Top()], halign = :right, "a.", fontsize = 20,
    font = :bold)
    lines!(ax, x, ss_line, color = :darkred, linestyle = :dash, label = "steady-state reaction rate",
        linewidth = 2.5)
    text!(ax, L"r_{ss} =\frac{k_{dec}}{Y_A} \left( \frac{\alpha_E}{k_{dec} (1/Y_D - \eta)} - K_B \right) " ,position=(maximum(x)*0.0, rate_ss.*1e10),
    align = (:left, :bottom), fontsize = 10)
    lines = lines!(ax, x, -1e10.*ca_rate_fd(sol.u[1][:,1], sol.u[1][:,2], sol.u[1][:,3], μₘ, Ka, Kd, Ya), color = (scale[1],0.6), label = "Time: $(sol.t[1] / 86400) days",
        linewidth = 2.9)
    for i in 1:length(sol.t)
        lines = lines!(ax, x, -1e10.*ca_rate_fd(sol.u[i][:,1], sol.u[i][:,2], sol.u[i][:,3], μₘ, Ka, Kd, Ya), color = (scale[i],0.6), label = "Time: $(sol.t[i] / 86400) days",
        linewidth = 2.9)
    end
    ax2 = Axis(fig[2,1:4], xlabel = "x [m]", xlabelsize = 15,
        xticklabelsize = 14,
        ylabel = "B [mol L⁻¹]", ylabelsize = 15,
        yticklabelsize = 14)
    ylims!(ax2, -1e-9, 2)
    lines!(ax2, x, bss_line, color = :darkred, linestyle = :dash, label = "Steady-state biomass concentration",
        linewidth = 2.5)
    text!(ax2, L"B_{ss} = \frac{\alpha_E}{k_{dec}(1/Y_D - \eta)} - K_B" ,position=(maximum(x)*0.03, Bss.*1e4),
    align = (:left, :bottom), fontsize = 10)
    l = lines = lines!(ax2, x, sol.u[1][:,3].*1e4, color = (scale[1],0.6), label = "Time: $(sol.t[1] / 86400) days",
        linewidth = 2.9)
    for i in 1:length(sol.t)
        lines = lines!(ax2, x, sol.u[i][:,3].*1e4, color = (scale[i],0.6), label = "Time: $(sol.t[i] / 86400) days",
        linewidth = 2.9)
    end
    Label(fig[2, 1:4, Top()], halign = :left, L"\times 10^{-4}")
    # add the scale bar
    Label(fig[2, 1:4, Top()], halign = :right, "b.", fontsize = 20,
    font = :bold)
    Colorbar(fig[1:2, 5], limits = (minimum(sol.t), maximum(sol.t))./86400, colormap=scale, label = "Time [days]", labelsize = 15, ticklabelsize = 14)
    fig
    save(plotsdir("reactBss.svg"), fig)
    save(plotsdir("reactBss.png"), fig, px_per_unit = 1200/96)
end

with_theme(theme_latexfonts()) do
    fig = Figure(size = (504, 350))
    ax2 = Axis(fig[1,1:4], xlabel = "PV [m]", xlabelsize = 15,
        xticklabelsize = 14,
        ylabel = "B [mol L⁻¹]", ylabelsize = 15,
        yticklabelsize = 14)
    ylims!(ax2, -1e-9, 2)
    bss_line = Bss .* ones(size(sol.t,1)).*1e4
    lines!(ax2, pore_volume.(sol.t), bss_line, color = :darkred, linestyle = :dash, label = "Steady-state biomass concentration",
        linewidth = 2.5)
    text!(ax2, L"B_{ss} = \frac{\alpha_E}{k_{dec}(1/Y_D - \eta)} - K_B" ,position=(maximum(x)*0.7, Bss.*1e4+0.05),
    align = (:left, :bottom), fontsize = 13)
    b_mid = [sol.u[i][Int(floor(size(u0,1)/2)),3] for i in 1:length(sol.t)]
    lines = lines!(ax2, pore_volume.(sol.t), b_mid.*1e4, color = glaucous, label = "B at x = 5 m",
        linewidth = 2.9)
    Label(fig[1, 1:4, Top()], halign = :left, L"\times 10^{-4}")
    fig
    save(plotsdir("Bss.svg"), fig)
    save(plotsdir("Bss.png"), fig, px_per_unit = 1200/96)
end



tspan = (0.0, 100*365*86400) # [s] 2 years
# creating the problem
prob = remake(prob, tspan = tspan)
# solving the problem
sol = solve(prob, Rosenbrock23(), saveat=86400, reltol = 1e-9, abstol = 1e-12)

with_theme(theme_latexfonts()) do
    fig = Figure( size = (504, 350))
    ax = Axis(fig[1, 1], xlabel = "x [m]", xlabelsize = 15,
        xticklabelsize = 18,
        ylabel = L"$C_A$ [mol L^{-1}]", ylabelsize = 15,
        yticklabelsize = 18)
    ax.xticks = 0:2:10
    ax.yticks = 0:2:6
    ylims!(ax, -1e-2, 6.3)
    
    half_sat_line = Ka .* ones(size(x,1))*1e4
    ca_min_line = ca_min .* ones(size(x,1))*1e4
    
    
    lines!(ax, x, half_sat_line, color = :black, linestyle = :dash,
        linewidth = 3.5)
    text!(ax, L"K_A" ,position=(maximum(x)*0.3, 1e4*(Ka+0.1e-4)), fontsize = 20, font="CMU Serif Bold")
    lines!(ax, x, ca_min_line, color = :darkblue, linestyle = :dash,
        linewidth = 3.5)
    text!(ax, L"C_{A,min}" ,position=(maximum(x)*0.3, 1e4*(ca_min+0.1e-4)), fontsize = 20)
    lines = lines!(ax, x, 1e4.*sol.u[1][:,1], color = (ashgray, 0.6), label = "Transient profiles",
        linewidth = 2.9)
    for i in 2:10:(length(sol.t)-1)
        lines3 = lines!(ax, x, 1e4.*sol.u[i][:, 1], color = (ashgray, 0.6),
        linewidth = 2.9)
    end
    lastline = lines!(ax, x, 1e4.*sol.u[end][:, 1], color = glaucous, label = "Profile at $(pore_volume(sol.t[end])) PV",
        linewidth = 2.9)
    xl = 0:1:L
    ss_line = cₐ .- rate_ss/v .* xl
    lines!(ax, xl[ss_line .>= ca_min], 1e4.*ss_line[ss_line .>= ca_min], color = :darkred, linestyle = :dash, label = "Advection steady-state prediction",
        linewidth = 2.1)
    Label(fig[1, 1, Top()], halign = :left, L"\times 10^{-4}")
    text!(ax, L"C_A = C_A^{in} - r_{ss}/v \times x" ,position=(maximum(x)*0.25, 1e4.*(cₐ .- rate_ss/v .* maximum(x)*0.25)-1.),
        align = (:left, :bottom), rotation = atan(-1e4*rate_ss/v,1), fontsize = 23)
    axislegend(position = :rt)
    save(plotsdir("ca_animation.svg"), fig)
    save(plotsdir("ca_animation.png"), fig, px_per_unit = 1200/96)
end

# Plotting the steady state concentration of bacteria

# Assuming Ca, Cd, Cd_min, and Ca_min are defined
Ca = 10 .^(range(log10(6.5e-5), stop=log10(1e-2), length=200))
Fk = k_dec/μₘ
A = Fk .* (Ca .+ Ka)./Ca
Cd = A .* Kd ./ (1 .- A)


with_theme(theme_latexfonts()) do
    xlow = 10. ^(-4.5)
    xhigh = 10. ^(-1.8)
    ylow = 10. ^(-7)
    yhigh = 10. ^(-1.8)
    fig = Figure(size = (504, 350))
    ax = Axis(fig[1, 1], xlabel = L"$C_A$ $[mol\, L^{-1}]$", ylabel = L"$C_D$ $[mol\, L^{-1}]$",
    xlabelsize = 15, ylabelsize = 15,
    xticklabelsize = 13, yticklabelsize = 13,
    limits = (xlow, xhigh, ylow, yhigh))
    # Plot Ca x Cd at steady-state
    lines!(ax, Ca, Cd, label = L"$B_{ss}$ $C_A$ x $C_D$ line",
    linewidth = 3.6, color = "steelblue")
    ct = 10. ^(-3.8)
    a = Fk .* (ct .+ Ka)./ct
    cdt = a .* Kd ./ (1 .- a)
    # Plot Cd_min and Ca_min lines
    Cd_line = range(cd_min, stop=1e-2, length=50)
    Ca_line = range(ca_min, stop=1e-2, length=50)
    lines!(ax, xlow:1e-7:xhigh, cd_min .* ones(length(xlow:1e-7:xhigh)), linestyle = (:dash,:loose), color = :black, linewidth = 2.5)
    lines!(ax, ca_min .* ones(length(ylow:1e-7:yhigh)), ylow:1e-7:yhigh, linestyle = (:dashdot,:loose), color = :black, linewidth = 2.5)
    text!(ax, L"C_{D,min}", position=(10. ^-4, cd_min), fontsize = 27, color = :black, align = (:left, :bottom), justification = :right)
    text!(ax, L"C_{A,min}", position=(ca_min, 10. ^-6.3), fontsize = 27, color = :black,
     align = (:left, :bottom), justification = :right, rotation = deg2rad(90))
    # Set axis properties
    reset_limits!(ax)
    # axislegend(ax, position = :rt)

    ax.xscale = CairoMakie.log10
    ax.yscale = CairoMakie.log10
    save(plotsdir("cacd.svg"), fig)
    save(plotsdir("cacd.png"), fig, px_per_unit = 1200/96)
end

