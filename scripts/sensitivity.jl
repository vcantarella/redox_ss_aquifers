using DrWatson
@quickactivate "redox_steady_state_aquifers"

using Symbolics
using CairoMakie
using DifferentialEquations
using LinearSolve
using DataFrames

println(
"""
We have the expression for the steady-state biomass and steady-state electron acceptor reaction rates. Finally we want to compute the sensitivity of the involved parameters
    to the expressions.
"""
)

# Define the variables

@variables α k_dec Yd η Kb Ya rss0 bss0

Bss_expr =  α/(k_dec*(1/Yd-η)) - Kb

rss_expr =  α/(Ya*(1/Yd-η))-Kb*k_dec/Ya

rel_rss_expr = (rss_expr-rss0)/rss0
rel_bss_expr = (Bss_expr-bss0)/bss0
# Compute the sensitivity

sensitivity_Bss = Symbolics.gradient(Bss_expr, [α, k_dec, Yd, η, Kb, Ya])
sensitivity_rss = Symbolics.gradient(rss_expr, [α, k_dec, Yd, η, Kb, Ya])
sensitivity_bss_rel = Symbolics.gradient(rel_bss_expr, [α, k_dec, Yd, η, Kb, Ya, bss0])
sensitivity_rss_rel = Symbolics.gradient(rel_rss_expr, [α, k_dec, Yd, η, Kb, Ya, rss0])

bss_f = eval(build_function(sensitivity_Bss, [α, k_dec, Yd, η, Kb, Ya])[1])
rss_f = eval(build_function(sensitivity_rss, [α, k_dec, Yd, η, Kb, Ya])[1])
bss_rel_sen = eval(build_function(sensitivity_bss_rel, [α, k_dec, Yd, η, Kb, Ya, bss0])[1])
rss_rel_sen = eval(build_function(sensitivity_rss_rel, [α, k_dec, Yd, η, Kb, Ya, rss0])[1])

rss_fun = eval(build_function(rss_expr, [α, k_dec, Yd, η, Kb, Ya]))
bss_fun = eval(build_function(Bss_expr, [α, k_dec, Yd, η, Kb, Ya]))
# defining the parameter values
# Defining the parameters
α = 8e-10 #[mol L⁻¹ s⁻¹]
Kb = 5e-5 #[mol L⁻¹] (equivalent water concentration)
Ka = 5e-4 #[mol L⁻¹]
Yd = 0.3 #[mol C_biomass/mol CD_substrate]
γa = 1/25 # stoichiometric coefficient of e-acceptor in the anabolic reaction
γc = 4/5 # stoichiometric coefficient of e-acceptor in the catabolic reaction
Ya = 1/(γa+(1/Yd-1)*γc) # [mol C_biomass/mol CA_substrate]
k_dec = 1.15e-6 #[s⁻¹]
Kd = 1e-6 #[mol L⁻¹]
η = 0.1 # [mol CD_substrate/mol C_biomass]

bss_sen = bss_f([α, k_dec, Yd, η, Kb, Ya])
rss_sen = rss_f([α, k_dec, Yd, η, Kb, Ya])

rss = rss_fun([α, k_dec, Yd, η, Kb, Ya])
bss = bss_fun([α, k_dec, Yd, η, Kb, Ya])

bss_rel = bss_rel_sen([α, k_dec, Yd, η, Kb, Ya, bss])
rss_rel = rss_rel_sen([α, k_dec, Yd, η, Kb, Ya, rss])
# Impact of the model sensitivity to the time to model outcomes:
no30 = 80/62*1e-3
upper_params = 1.1 .* [α, k_dec, Yd, η, Kb, Ya]
lower_params = 0.9 .* [α, k_dec, Yd, η, Kb, Ya]
df = DataFrame(
    Parameter = ["α", "k_dec", "Yd", "η", "Kb", "Ya"],
    Value = [α, k_dec, Yd, η, Kb, Ya],
    Bss = bss,
    Bss_sensitivity = bss_sen,
    Rss = rss,
    Rss_sensitivity = rss_sen,
    Upper = upper_params,
    Lower = lower_params,
    Bss_lower = zeros(Float64, 6),
    Bss_upper = zeros(Float64, 6),
    Rss_lower = zeros(Float64, 6),
    Rss_upper = zeros(Float64, 6),
)

for i in eachindex(df[!, :Parameter])
    parameters = [α, k_dec, Yd, η, Kb, Ya]
    param_low = copy(parameters)
    param_low[i] = lower_params[i]
    param_high = copy(parameters)
    param_high[i] = upper_params[i]
    df[i, :Bss_lower] = bss_fun(param_low)
    df[i, :Bss_upper] = bss_fun(param_high)
    df[i, :Rss_lower] = rss_fun(param_low)
    df[i, :Rss_upper] = rss_fun(param_high)
end
no30/rss


# Plot the sensitivities in two horizontal bar plots, one for the bss and one for the rss
# Parameter names
# Sort indices by magnitude of bss sensitivity
param_names = [L"\alpha", L"k_{dec}", L"Y_D", L"\eta", L"K_B", L"Y_A"]
sorted_idx_bss = sortperm(abs.(bss_sen), rev=false)
sorted_idx_rss = sortperm(abs.(rss_sen), rev=false)
params = [α, k_dec, Yd, η, Kb, Ya]

with_theme(theme_latexfonts()) do
    fig = Figure(size=(800, 400))
    
    # First subplot for bss sensitivities
    ax1 = Axis(fig[1, 1],
        title = L"B_{ss} \mathrm{~sensitivity}",
        xlabel = L"\text{Parameters}\, p",
        ylabel = L" \frac{\partial B_{ss}}{\partial p}\,\frac{p}{B_{ss}}",
        xticks = (1:length(param_names), param_names[sorted_idx_bss]),
        yticks = -1.2:0.4:1.2,
        )
    
    # Second subplot for rss sensitivities
    ticks = [0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-0]
    ax2 = Axis(fig[1, 2],
        title = L"r_{ss} \mathrm{~sensitivity}",
        xlabel = L"\text{Parameters}\, p", 
        ylabel = L"\frac{\partial r_{ss}}{\partial p}\,\frac{p}{r_{ss}}",
        xticks = (1:length(param_names), param_names[sorted_idx_bss]),
        yticks = -1.2:0.4:1.2,
        )
    linkyaxes!(ax1, ax2)
    strokecolor = :black
    strokewidth = 0.6
    # Create horizontal bar plots
    barplot!(ax1, 1:length(param_names), bss_sen[sorted_idx_bss].*params[sorted_idx_bss]./bss,
        strokecolor = strokecolor, strokewidth = strokewidth)
    barplot!(ax2, 1:length(param_names), rss_sen[sorted_idx_bss].*params[sorted_idx_bss]./rss,
        strokecolor = strokecolor, strokewidth = strokewidth)
    
    fig
    save(plotsdir("sensitivity_bss_rssv2.png"), fig, px_per_unit = 1200/96)
    save(plotsdir("sensitivity_bss_rssv2.svg"), fig)
end


with_theme(theme_latexfonts()) do
    fig = Figure(size=(800, 400))
    #tickfont = 11
    
    # First subplot for bss sensitivities
    ax1 = Axis(fig[1, 1],
        title = L"B_{ss}",
        xlabel = L"\text{Parameters}\, p",
        ylabel = L"\mathrm{Change\, in}\, B_{ss} \, [-]",
        xticks = (1:length(param_names), param_names[sorted_idx_bss]),
        #xticks = [0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6],
        #xticklabelsize = tickfont,
        #yscale = Makie.log10,
        #xtickformat = values -> [L"10^{%$(log10(value))}" for value in values]
        )
    
    # Second subplot for rss sensitivities
    ticks = [0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-0]
    ax2 = Axis(fig[1, 2],
        title = L"r_{ss}",
        xlabel = L"\text{Parameters}\, p", 
        ylabel = L"\mathrm{Change\, in}\, r_{ss} \, [-]",
        xticks = (1:length(param_names), param_names[sorted_idx_bss]),
        #xticks = [0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-0],
        # yscale = Makie.log10,
        #xtickformat = values -> [L"10^{%$(log10(value))}" for value in values],
        )
    linkyaxes!(ax1, ax2)
    strokecolor = :black
    strokewidth = 0.6
    # Create horizontal bar plots
    barplot!(ax1, 1:length(param_names), (df[!, :Bss_upper][sorted_idx_bss].-bss)./bss,
        label="Parameters increased by 10%",  strokecolor = strokecolor, strokewidth = strokewidth)
    barplot!(ax2, 1:length(param_names), (df[!, :Rss_upper][sorted_idx_bss].-rss)./rss,
        label="Parameters increased by 10%",  strokecolor = strokecolor, strokewidth = strokewidth)
    barplot!(ax1, 1:length(param_names), (df[!, :Bss_lower][sorted_idx_bss].-bss)./bss,
        label="Parameters decreased by 10%", color = (:crimson, 0.7),  strokecolor = strokecolor, strokewidth = strokewidth)
    barplot!(ax2, 1:length(param_names), (df[!, :Rss_lower][sorted_idx_bss].-rss)./rss,
        label="Parameters increased by 10%", color = (:crimson, 0.7),  strokecolor = strokecolor, strokewidth = strokewidth)
    Legend(fig[2, 1:2],ax1, framevisible = false, orientation = :horizontal, halign = :center, valign = :top)
    resize_to_layout!(fig)
    fig
    save(plotsdir("sensitivity_bss_rss_relat.png"), fig, px_per_unit = 1200/96)
    save(plotsdir("sensitivity_bss_rss_relat.svg"), fig)
end


#----------------- Apply the sensitivity analysis to the model -----------------
println(
"""
We have seen that the most relevant parameter is the maximum hydrolysis rate. We will now apply the sensitivity analysis to the model.
"""
)

# Load the model
include(srcdir("ode_model.jl"))


## ---- normal parameters model ---------
# Parameters
nmob = 2
nsub = 4

# Defining the parameters
α = 8e-10 #[mol L⁻¹ s⁻¹]
α₀ = 3e-10
αᵤ = 8e-9
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
dx_high = 0.005 # [m]
dx_low = 0.1 # [m]
L = 10 # [m]
v = 0.5/86400 # [m/s]
x = 0:dx:L
x_low = 0:dx_low:50
x_high = 0:dx_high:1

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
rate_low!, ca_rate_fd_low= create_fixed_decay(v, D, dx_low, c_in, nmob)
rate_high!, ca_rate_fd_high= create_fixed_decay(v, D, dx_high, c_in, nmob)


#---- Model running: fixed decay+ maintenance-------
# Initial conditions
u0 = zeros(size(x,1), nsub)
u0_low = zeros(size(x_low,1), nsub)
u0_high = zeros(size(x_high,1), nsub)
du0 = u0
Bss = α/(k_dec*(1/Yd-η)) - Kb
Bss_low = α₀/(k_dec*(1/Yd-η)) - Kb
Bss_high = αᵤ/(k_dec*(1/Yd-η)) - Kb
b₀ = 0.1
rate_ss = α/(Ya*(1/Yd-η))-Kb*k_dec/Ya
rate_ss_low = α₀/(Ya*(1/Yd-η))-Kb*k_dec/Ya
rate_ss_high = αᵤ/(Ya*(1/Yd-η))-Kb*k_dec/Ya
u0[:,3] .= Bss*b₀
u0_low[:,3] .= Bss_low*b₀
u0_high[:,3] .= Bss_high*b₀
Cs₀ = 1 # mol/kg_sed
ρₛ = 2.65 # [g/cm³]/[kg/L]
ϕ = 0.3
cf = (1-ϕ)*ρₛ/ϕ # [conversion from molC/kg_sed to molC/L_water]
Cs₀ = cf*Cs₀
u0[:,4] .= Cs₀
u0_low[:,4] .= Cs₀
u0_high[:,4] .= Cs₀

# parameter vector and table:
p = [α, Kb, μₘ, Ka, Kd, Ya, Yd, k_dec, Ks, η]
p_low = [α₀, Kb, μₘ, Ka, Kd, Ya, Yd, k_dec, Ks, η]
p_high = [αᵤ, Kb, μₘ, Ka, Kd, Ya, Yd, k_dec, Ks, η]
ref = ["a", "b", "c", "d", "d", "c", "c", "e", missing, missing,]
description = ["Maximum hydrolysis rate", "Half-saturation constant of biomass",
"Maximum growth rate", "Half-saturation constant of electron acceptor", "Half-saturation constant of electron donor",
"Yield of biomass on electron acceptor", "Yield of biomass on electron donor", "Decay rate constant",
"Half-saturation constant of sediment organic carbon", "proportion of liable biomass decayed"]
symbol = [L"\alpha", L"K_B", L"\mu_{max}", L"K_A", L"K_D", L"Y_A", L"Y_D", L"k_{dec}", L"K_S", L"\eta"]
# steady-state Bss in cells/g_sed
smass_bio = 24.6 # [g/cmolbio]
cells_pm = Bss * smass_bio * ϕ/ (100*1e-15) # [cells/L]
cells_g = cells_pm/(1-ϕ)/(ρₛ*1000) # [cells/g_sed]
# Time span
tspan = (0.0, 5*365*86400) # [s] 2 years
# creating the problem
jac_sparsity = Symbolics.jacobian_sparsity((du, u) -> fixed_decay!(du, u, p, 1),
    du0, u0) # add the sparsity pattern to speed up the solution
fixed_rhs! = ODEFunction(fixed_decay!, jac_prototype=jac_sparsity)
jac_sparsity_low = Symbolics.jacobian_sparsity((du, u) -> rate_low!(du, u, p_low, 1),
    copy(u0_low), u0_low) # add the sparsity pattern to speed up the solution
fixed_low! = ODEFunction(rate_low!, jac_prototype=jac_sparsity_low)
jac_sparsity_high = Symbolics.jacobian_sparsity((du, u) -> rate_high!(du, u, p_high, 1),
    copy(u0_high), u0_high) # add the sparsity pattern to speed up the solution
fixed_high! = ODEFunction(rate_high!, jac_prototype=jac_sparsity_high)
prob = ODEProblem(fixed_rhs!, u0, tspan, p)
prob_low = ODEProblem(fixed_low!, u0_low, tspan, p_low)
prob_high = ODEProblem(fixed_high!, u0_high, tspan, p_high)
# solving the problem
sol = solve(prob, Rosenbrock23(linsolve=KrylovJL_GMRES()), saveat=86400, reltol = 1e-8, abstol = 1e-8)
println("Model solved")
sol_high = solve(prob_high, Rosenbrock23(linsolve=KrylovJL_GMRES()), saveat=86400, reltol = 1e-8, abstol = 1e-8)
println("high rate model solved")
sol_low = solve(prob_low, Rosenbrock23(linsolve=KrylovJL_GMRES()), saveat=86400, reltol = 1e-8, abstol = 1e-8)
println("low rate model solved")

function pore_volume(t)
    return v*ϕ*t/L
end
# Assuming Ca, Cd, Cd_min, and Ca_min are defined
Ca = 10 .^(range(log10(6.5e-5), stop=log10(1e-2), length=200))
Fk = k_dec/μₘ
A = Fk .* (Ca .+ Ka)./Ca
Cd = A .* Kd ./ (1 .- A)
crimson = "#DC143C"
ashgray = "#B2BEB5"
pvs = pore_volume.(sol.t) # calculate the pore volumes
inv_pv = [0.1,0.3, 0.5, 1, 2, 10, 20]
pvs_index1 = [findfirst(pvs .>= i) for i in inv_pv]
pvs = pvs[pvs_index1]
color_values = (pvs)
scale = cgrad(:starrynight, color_values, rev = true, scale = :exp,
    categorical = false)
final_color = "#0B192C"
pvs_max = maximum(pvs)
with_theme(theme_latexfonts()) do
    labelsize = 11
    ticksize = 10
    line_text = 10
    minor_line = 1.5
    major_line = 2.6
    title = 12
    width = 225
    height = 8/10*width
    fig = Figure( size = (10*96, 9*96))
    ax = Axis(fig[1:3, 1:5], xlabelsize = labelsize,
        xticklabelsize = ticksize,
        ylabel = L"$C_A$ [mol L^{-1}]", ylabelsize = labelsize,
        yticklabelsize = ticksize,
        xgridwidth = 0.5, ygridwidth = 0.5,
        width = width, height = height)
    ax.xticks = 0:2:10
    ax.yticks = 0:2:6
    ylims!(ax, -1e-2, 6.3)
    half_sat_line = Ka .* ones(size(x,1))*1e4
    ca_min_line = ca_min .* ones(size(x,1))*1e4
    bss_line = Bss .* ones(size(x,1)).*1e4
    
    lines!(ax, x, half_sat_line, color = :black, linestyle = :dash,
        linewidth = major_line)
    text!(ax, L"K_A" ,position=(maximum(x)*0.3, 1e4*(Ka+0.1e-4)), fontsize = line_text, font="CMU Serif Bold")
    lines!(ax, x, ca_min_line, color = :darkblue, linestyle = :dash,
        linewidth = major_line)
    text!(ax, L"C_{A,min}" ,position=(maximum(x)*0.3, 1e4*(ca_min+0.1e-4)), fontsize = line_text)
    lines = lines!(ax, x, 1e4.*sol.u[1][:,1], color = (ashgray, 0.7), # label = "Transient profiles",
        linewidth = minor_line)
    k = 1
    for i in pvs_index1# 2:10:(length(sol.t)-1)
        k = findfirst(pvs_max.*scale.values .>= pore_volume(sol.t[i]))
        lines3 = lines!(ax, x, 1e4.*sol.u[i][:, 1], color = (scale[k], 0.9),
        linewidth = minor_line)
        k+=1
    end
    lastline = lines!(ax, x, 1e4.*sol.u[end][:, 1], color = final_color, label = "Profile at $(pore_volume(sol.t[end])) PV",
        linewidth = major_line)
    xl = 0:0.01:L
    ss_line = cₐ .- rate_ss/v .* xl
    ss_line = collect(ss_line)
    ss_line[ss_line .< ca_min] .= ca_min
    A_l = Fk .* (ss_line .+ Ka)./ss_line
    cd_line = A_l .* Kd ./ (1 .- A_l)
    lines!(ax, xl, 1e4.*ss_line, color = crimson, linestyle = :dash, label = "Analytical prediction",
        linewidth = major_line)
    Label(fig[1:3, 1:5, Top()], halign = :left, L"\times 10^{-4}", fontsize = 10)
    Label(fig[1:3, 1:5, Top()], halign = :center, "a. Electron acceptor, α = 8 x 10⁻¹⁰", fontsize = title)
    # axislegend(position = :rt)

    ax2 = Axis(fig[1:3, 6:10], xlabelsize = labelsize,
        xticklabelsize = ticksize,
        ylabel = L"$C_A$ [mol L^{-1}]", ylabelsize = labelsize,
        yticklabelsize = ticksize,
        xgridwidth = 0.5, ygridwidth = 0.5,
        width = width, height = height)
    ax2.xticks = 0:10:50
    ax2.yticks = 0:2:6
    ylims!(ax2, -1e-2, 6.3)
    half_sat_line = Ka .* ones(size(x_low,1))*1e4
    ca_min_line = ca_min .* ones(size(x_low,1))*1e4
    
    lines!(ax2, x_low, half_sat_line, color = :black, linestyle = :dash,
        linewidth = major_line)
    text!(ax2, L"K_A" ,position=(maximum(x)*0.3, 1e4*(Ka+0.1e-4)), fontsize = line_text, font="CMU Serif Bold")
    lines!(ax2, x_low, ca_min_line, color = :darkblue, linestyle = :dash,
        linewidth = major_line)
    text!(ax2, L"C_{A,min}" ,position=(maximum(x)*0.3, 1e4*(ca_min+0.1e-4)), fontsize = line_text)
    lines = lines!(ax2, x_low, 1e4.*sol_low.u[1][:,1], color = (ashgray, 0.7), # label = "Transient profiles",
        linewidth = minor_line)
    k = 1
    for i in pvs_index1# 2:10:(length(sol.t)-1)
        k = findfirst(pvs_max.*scale.values .>= pore_volume(sol.t[i]))
        lines3 = lines!(ax2, x_low, 1e4.*sol_low.u[i][:, 1], color = (scale[k], 0.9),
        linewidth = minor_line)
        k+=1
    end
    lastline = lines!(ax2, x_low, 1e4.*sol_low.u[end][:, 1], color = final_color, label = "Profile at $(pore_volume(sol.t[end])) PV",
        linewidth = major_line)
    xl = 0:0.01:x_low[end]
    ss_line_low = cₐ .- rate_ss_low/v .* xl
    ss_line_low = collect(ss_line_low)
    ss_line_low[ss_line_low .< ca_min] .= ca_min
    A_l = Fk .* (ss_line_low .+ Ka)./ss_line_low
    cd_line = A_l .* Kd ./ (1 .- A_l)
    lines!(ax2, xl, 1e4.*ss_line_low, color = crimson, linestyle = :dash, label = "Analytical prediction",
        linewidth = major_line)
    Label(fig[1:3, 6:10, Top()], halign = :left, L"\times 10^{-4}", fontsize = 10)
    Label(fig[1:3, 6:10, Top()], halign = :center,  "b. Electron acceptor, α = 3 x 10⁻¹⁰", fontsize = title)
    # axislegend(position = :rt)

    ax3 = Axis(fig[1:3, 11:15], xlabelsize = labelsize,
        xticklabelsize = ticksize,
        ylabel = L"$C_A$ [mol L^{-1}]", ylabelsize = labelsize,
        yticklabelsize = ticksize,
        xgridwidth = 0.5, ygridwidth = 0.5,
        width = width, height = height)
    ax3.xticks = 0:.2:1
    ax3.yticks = 0:2:6
    ylims!(ax3, -1e-2, 6.3)
    half_sat_line = Ka .* ones(size(x_high,1))*1e4
    ca_min_line = ca_min .* ones(size(x_high,1))*1e4
    
    lines!(ax3, x_high, half_sat_line, color = :black, linestyle = :dash,
        linewidth = major_line)
    text!(ax3, L"K_A" ,position=(maximum(x_high)*0.3, 1e4*(Ka+0.1e-4)), fontsize = line_text, font="CMU Serif Bold")
    lines!(ax3, x_high, ca_min_line, color = :darkblue, linestyle = :dash,
        linewidth = major_line)
    text!(ax3, L"C_{A,min}" ,position=(maximum(x_high)*0.3, 1e4*(ca_min+0.1e-4)), fontsize = line_text)
    lines = lines!(ax3, x_high, 1e4.*sol_high.u[1][:,1], color = (ashgray, 0.7), # label = "Transient profiles",
        linewidth = minor_line)
    k = 1
    for i in pvs_index1# 2:10:(length(sol.t)-1)
        k = findfirst(pvs_max.*scale.values .>= pore_volume(sol.t[i]))
        lines3 = lines!(ax3, x_high, 1e4.*sol_high.u[i][:, 1], color = (scale[k], 0.9),
        linewidth = minor_line)
        k+=1
    end
    lastline = lines!(ax3, x_high, 1e4.*sol_high.u[end][:, 1], color = final_color, label = "Profile at $(pore_volume(sol.t[end])) PV",
        linewidth = major_line)
    xl = 0:0.01:x_high[end]
    ss_line_high = cₐ .- rate_ss_high/v .* xl
    ss_line_high = collect(ss_line_high)
    ss_line_high[ss_line_high .< ca_min] .= ca_min
    A_l = Fk .* (ss_line_high .+ Ka)./ss_line_high
    cd_line = A_l .* Kd ./ (1 .- A_l)
    lines!(ax3, xl, 1e4.*ss_line_high, color = crimson, linestyle = :dash, label = "Analytical prediction",
        linewidth = major_line)
    Label(fig[1:3, 11:15, Top()], halign = :left, L"\times 10^{-4}", fontsize = 10)
    Label(fig[1:3, 11:15, Top()], halign = :center, "c. Electron acceptor α = 8 x 10⁻⁹", fontsize = title)
    # axislegend(position = :rt)

    ax4 = Axis(fig[4:6, 1:5], xlabelsize = labelsize,
        xticklabelsize = ticksize,
        ylabel = L"$r_{C_A}$ [mol L⁻¹ s⁻¹]", ylabelsize = labelsize,
        yticklabelsize = ticksize,
        xgridwidth = 0.5, ygridwidth = 0.5,
        width = width, height = height)
    ax4.xticks = 0:2:10
    ax4.yticks = 0:1:5
    ylims!(ax4, -1e-9, 5.2)
    xl = 0:0.01:L
    ss_r_line = rate_ss .* ones(size(xl,1)).*1e10
    ss_r_line[ss_line .<= ca_min] .= 0
    bss_line = Bss .* ones(size(xl,1)).*1e4
    bss_line[ss_line .<= ca_min] .= 0
    Label(fig[4:6, 1:5, Top()], halign = :left, L"\times 10^{-10}", fontsize = 10)
    Label(fig[4:6, 1:5, Top()], halign = :center, "d. Reaction rate α = 8 x 10⁻¹⁰", fontsize = title)
    lines = lines!(ax4, x, -1e10.*ca_rate_fd(sol.u[1][:,1], sol.u[1][:,2], sol.u[1][:,3], μₘ, Ka, Kd, Ya), color = (ashgray, 0.0), #label = "Transient profiles",
        linewidth = minor_line)
    k = 1
    for i in pvs_index1
        k = findfirst(pvs_max.*scale.values .>= pore_volume(sol.t[i]))
        lines = lines!(ax4, x, -1e10.*ca_rate_fd(sol.u[i][:,1], sol.u[i][:,2], sol.u[i][:,3], μₘ, Ka, Kd, Ya), color = (scale[k], 0.9), label = "PV: $(round(pore_volume(sol.t[i]), digits = 1))",
        linewidth = minor_line)
        k+=1
    end
    last_line = lines!(ax4, x, -1e10.*ca_rate_fd(sol.u[end][:,1], sol.u[end][:,2], sol.u[end][:,3], μₘ, Ka, Kd, Ya),
     color = final_color, # label = "Transient Profiles",
    linewidth = major_line)
    lines!(ax4, xl, ss_r_line, color = crimson, linestyle = :dash, #label = "steady-state reaction rate",
        linewidth = major_line)

    ax5 = Axis(fig[4:6, 6:10], xlabelsize = labelsize,
    xticklabelsize = ticksize,
    ylabel = L"$r_{C_A}$ [mol L⁻¹ s⁻¹]", ylabelsize = labelsize,
    yticklabelsize = ticksize,
    xgridwidth = 0.5, ygridwidth = 0.5,
    width = width, height = height)
    ax5.xticks = 0:10:50
    ax5.yticks = 0:1:5
    ylims!(ax5, -1e-9, 5.2)
    xl_low = 0:0.01:x_low[end]
    ss_r_line_low = rate_ss_low .* ones(size(xl_low,1)).*1e10
    ss_r_line_low[ss_line_low .<= ca_min] .= 0
    bss_line_low = Bss_low .* ones(size(xl_low,1)).*1e4
    bss_line_low[ss_line_low .<= ca_min] .= 0
    Label(fig[4:6, 6:10, Top()], halign = :left, L"\times 10^{-10}", fontsize = 10)
    Label(fig[4:6, 6:10, Top()], halign = :center, "e. Reaction rate α = 3 x 10⁻¹⁰", fontsize = title)
    lines = lines!(ax5, x_low, -1e10.*ca_rate_fd(sol_low.u[1][:,1], sol_low.u[1][:,2], sol_low.u[1][:,3], μₘ, Ka, Kd, Ya), color = (ashgray, 0.0), label = "Transient profiles",
        linewidth = minor_line)
    k = 1
    for i in pvs_index1
        k = findfirst(pvs_max.*scale.values .>= pore_volume(sol.t[i]))
        lines = lines!(ax5, x_low, -1e10.*ca_rate_fd(sol_low.u[i][:,1], sol_low.u[i][:,2], sol_low.u[i][:,3], μₘ, Ka, Kd, Ya), color = (scale[k], 0.9), label = "Transient Profiles",
        linewidth = minor_line)
        k+=1
    end
    last_line = lines!(ax5, x_low, -1e10.*ca_rate_fd(sol_low.u[end][:,1], sol_low.u[end][:,2], sol_low.u[end][:,3], μₘ, Ka, Kd, Ya),
     color = final_color, # label = "Transient Profiles",
    linewidth = major_line)
    lines!(ax5, xl_low, ss_r_line_low, color = crimson, linestyle = :dash, label = "steady-state reaction rate",
        linewidth = major_line)
        
    ax6 = Axis(fig[4:6, 11:15], xlabelsize = labelsize,
    xticklabelsize = ticksize,
    ylabel = L"$r_{C_A}$ [mol L⁻¹ s⁻¹]", ylabelsize = labelsize,
    yticklabelsize = ticksize,
    xgridwidth = 0.5, ygridwidth = 0.5,
    width = width, height = height)
    ax6.xticks = 0:.2:1
    ax6.yticks = 0:1:5
    ylims!(ax6, -1e-9, 5.2)
    xl_high = 0:0.01:x_high[end]
    ss_r_line_high = rate_ss_high .* ones(size(xl_high,1)).*1e9
    ss_r_line_high[ss_line_high .<= ca_min] .= 0
    bss_line_high = Bss_high .* ones(size(xl_high,1)).*1e4
    bss_line_high[ss_line_high .<= ca_min] .= 0
    Label(fig[4:6, 11:15, Top()], halign = :left, L"\times 10^{-10}", fontsize = 10)
    Label(fig[4:6, 11:15, Top()], halign = :center, "f. Reaction rate α = 8 x 10⁻⁹", fontsize = title)
    lines = lines!(ax6, x_high, -1e10.*ca_rate_fd(sol_high.u[1][:,1], sol_high.u[1][:,2], sol_high.u[1][:,3], μₘ, Ka, Kd, Ya), color = (ashgray, 0.0), label = "Transient profiles",
        linewidth = minor_line)
    k = 1
    for i in pvs_index1
        k = findfirst(pvs_max.*scale.values .>= pore_volume(sol.t[i]))
        lines = lines!(ax6, x_high, -1e9.*ca_rate_fd(sol_high.u[i][:,1], sol_high.u[i][:,2], sol_high.u[i][:,3], μₘ, Ka, Kd, Ya), color = (scale[k], 0.9), label = "Transient Profiles",
        linewidth = minor_line)
        k+=1
    end
    last_line = lines!(ax6, x_high, -1e9.*ca_rate_fd(sol_high.u[end][:,1], sol_high.u[end][:,2], sol_high.u[end][:,3], μₘ, Ka, Kd, Ya),
        color = final_color, # label = "Transient Profiles",
    linewidth = major_line)
    lines!(ax6, xl_high, ss_r_line_high, color = crimson, linestyle = :dash, label = "steady-state reaction rate",
        linewidth = major_line)

    ax7 = Axis(fig[7:9,1:5], xlabel = "x [m]", xlabelsize = labelsize,
        xticklabelsize = ticksize,
        ylabel = "B [mol L⁻¹]", ylabelsize = labelsize,
        yticklabelsize = ticksize,
        xgridwidth = 0.5, ygridwidth = 0.5,
        width = width, height = height)
    ax7.xticks = 0:2:10
    ax7.yticks = 0:1:5
    ylims!(ax7, -1e-9, 5.4)
    Label(fig[7:9,1:5, Top()], halign = :left, L"\times 10^{-4}", fontsize = 10)
    Label(fig[7:9,1:5, Top()], halign = :center, "g. Biomass α = 8 x 10⁻¹⁰", fontsize = title)
    l = lines = lines!(ax7, x, sol.u[1][:,3].*1e4, color = (ashgray, 0.0), # label = "Transient profiles",
        linewidth = minor_line)
    k = 1
    for i in pvs_index1
        k = findfirst(pvs_max.*scale.values .>= pore_volume(sol.t[i]))
        lines = lines!(ax7, x, sol.u[i][:,3].*1e4, color = (scale[k], 0.9), label = "$(round(pore_volume(sol.t[i]), digits = 1))",
        linewidth = minor_line)
        k+=1
    end
    last_line = lines!(ax7, x, sol.u[end][:,3].*1e4, color = final_color, # label = "Final Profile",
    linewidth = major_line)
    lines!(ax7, xl, bss_line, color = crimson, linestyle = :dash, # label = "Steady-state biomass concentration",
        linewidth = major_line)
    
    ax8 = Axis(fig[7:9,6:10], xlabel = "x [m]", xlabelsize = labelsize,
    xticklabelsize = ticksize,
    ylabel = "B [mol L⁻¹]", ylabelsize = labelsize,
    yticklabelsize = ticksize,
    xgridwidth = 0.5, ygridwidth = 0.5,
    width = width, height = height)
    ax8.xticks = 0:10:50
    ax8.yticks = 0:1:5
    ylims!(ax8, -1e-9, 5.4)
    Label(fig[7:9,6:10, Top()], halign = :left, L"\times 10^{-4}", fontsize = 10)
    Label(fig[7:9,6:10, Top()], halign = :center, "h. Biomass α = 3 x 10⁻¹⁰", fontsize = title)
    l = lines = lines!(ax8, x_low, sol_low.u[1][:,3].*1e4, color = (ashgray, 0.0), # label = "Transient profiles",
        linewidth = minor_line)
    k = 1
    for i in pvs_index1
        k = findfirst(pvs_max.*scale.values .>= pore_volume(sol.t[i]))
        lines = lines!(ax8, x_low, sol_low.u[i][:,3].*1e4, color = (scale[k], 0.9), label = "$(round(pore_volume(sol.t[i]), digits = 1))",
        linewidth = minor_line)
        k+=1
    end
    last_line = lines!(ax8, x_low, sol_low.u[end][:,3].*1e4, color = final_color, # label = "Final Profile",
    linewidth = major_line)
    lines!(ax8, xl_low, bss_line_low, color = crimson, linestyle = :dash, # label = "Steady-state biomass concentration",
        linewidth = major_line)
    
    ax9 = Axis(fig[7:9,11:15], xlabel = "x [m]", xlabelsize = labelsize,
    xticklabelsize = ticksize,
    ylabel = "B [mol L⁻¹]", ylabelsize = labelsize,
    yticklabelsize = ticksize,
    xgridwidth = 0.5, ygridwidth = 0.5,
    width = width, height = height)
    ax9.xticks = 0:0.2:1
    ax9.yticks = 0:5:25
    ylims!(ax9, -1e-9, 30)
    Label(fig[7:9,11:15, Top()], halign = :left, L"\times 10^{-4}", fontsize = 10)
    Label(fig[7:9,11:15, Top()], halign = :center, "i. Biomass α = 8 x 10⁻⁹", fontsize = title)
    l = lines = lines!(ax9, x_high, sol_high.u[1][:,3].*1e4, color = (ashgray, 0.0), # label = "Transient profiles",
        linewidth = minor_line)
    k = 1
    for i in pvs_index1
        k = findfirst(pvs_max.*scale.values .>= pore_volume(sol.t[i]))
        lines = lines!(ax9, x_high, sol_high.u[i][:,3].*1e4, color = (scale[k], 0.9), label = "$(round(pore_volume(sol.t[i]), digits = 1))",
        linewidth = minor_line)
        k+=1
    end
    last_line = lines!(ax9, x_high, sol_high.u[end][:,3].*1e4, color = final_color, # label = "Final Profile",
    linewidth = major_line)
    lines!(ax9, xl_high, bss_line_high, color = crimson, linestyle = :dash, # label = "Steady-state biomass concentration",
        linewidth = major_line)
    fig[10,5:10] = Legend(fig, ax, framevisible = false, fontsize = 10, orientation = :horizontal, halign = :center, valign = :top)
    fig[11,5:10] = Legend(fig[8,3:8], ax4, "Pore Volumes - Transient profiles", framevisible = false, fontsie = 10,
        orientation = :horizontal, halign = :center, valign = :top,)
    

    resize_to_layout!(fig)
    fig
    save(plotsdir("sensitivity_analysis.png"), fig, px_per_unit = 1200/96)
    save(plotsdir("sensitivity_analysis.svg"), fig)
    
end
