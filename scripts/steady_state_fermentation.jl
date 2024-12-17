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
D = k_dec2/μ₂*(c2_max+Kd)/c2_max
ca_min_from_c2max = D/(1-D)*Ka
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
ϕ = 0.3 # porosity [-]
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
    fig = Figure(size = (504, 350))
    ax = Axis(fig[1, 1], xlabel = "x [m]", xlabelsize = 15,
        xticklabelsize = 18,
        ylabel = L"$C_A$ [mol L⁻¹]", ylabelsize = 15,
        yticklabelsize = 18)
    ax.xticks = 0:2:10
    ax.yticks = 0:1:6
    ylims!(ax, -1e-9, 6.3)
    # axis with squared proportions
    # I want one unit from the x-axis to be 1 unit in the y-axis
    ax.aspect = DataAspect()
    ss_line = (cₐ .- rate_ss/v .* x).*1e4
    half_sat_line = Ka .* ones(size(x,1)).*1e4
    ca_min_line = ca_min_from_c2max .* ones(size(x,1)).*1e4
    lines!(ax, x, ss_line, color = :darkred, linestyle = :dash, label = "Avection dominated prediction",
        linewidth = 2.5)
    text!(ax, L"C_A = C_A^{in} - r_{ss}/v \times x" ,position=(maximum(x)*0.5, (cₐ .- rate_ss/v .* maximum(x)*0.5).*1e4),
    align = (:left, :bottom), rotation = atan(-rate_ss*1e4/v,1), fontsize = 15)
    lines!(ax, x, half_sat_line, color = :black, linestyle = :dash, label = "Half saturation constant",
        linewidth = 3.5)
    text!(ax, L"K_A" ,position=(maximum(x)*0.3, (Ka+0.1e-4).*1e4), fontsize = 15, font="CMU Serif Bold")
    lines!(ax, x, ca_min_line, color = :darkblue, linestyle = :dash, label = "Minimum concentration",
        linewidth = 3.5)
    text!(ax, L"C_A^{min}" ,position=(maximum(x)*0.3, (ca_min+0.1e-4).*1e4), fontsize = 15)
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
color_values = (sol.t)
scale = cgrad(:deep, color_values ./ maximum(color_values))
ashgray = "#B2BEB5"
glaucous = "#6082B6"


crimson = "#DC143C"
pvs = pore_volume.(sol.t) # calculate the pore volumes
pvs_index0 = findall(pvs .<= 3) # selecint the pore volumes to plot
pvs_index1 = pvs_index0[2:20:end]
pvs = pvs[pvs .<= 3]
pvs = pvs[2:20:end]
color_values = (pvs)
scale = cgrad(:imola10, color_values ./ maximum(color_values), rev = true,# scale = :exp,
    categorical = true)
final_color = "#0B192C"


with_theme(theme_latexfonts()) do
    labelsize = 11
    ticksize = 10
    line_text = 10
    minor_line = 1.5
    major_line = 2.6
    title = 12
    width = 225
    height = 8/10*width
    fig = Figure( size = (6.8*96, 9*96))
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
    ca_min_line = ca_min_from_c2max .* ones(size(x,1))*1e4
    bss_line1 = Bss1 .* ones(size(x,1)).*1e4
    bss_line2 = Bss2 .* ones(size(x,1)).*1e4
    
    lines!(ax, x, half_sat_line, color = :black, linestyle = :dash,
        linewidth = major_line)
    text!(ax, L"K_A" ,position=(maximum(x)*0.3, 1e4*(Ka+0.1e-4)), fontsize = line_text, font="CMU Serif Bold")
    lines!(ax, x, ca_min_line, color = :darkblue, linestyle = :dash,
        linewidth = major_line)
    text!(ax, L"C_{A,min}" ,position=(maximum(x)*0.3, 1e4*(ca_min_from_c2max+0.1e-4)), fontsize = line_text)
    lines = lines!(ax, x, 1e4.*sol.u[1][:,1], color = (ashgray, 0.7), label = "Transient profiles",
        linewidth = minor_line)
    k = 1
    for i in pvs_index1# 2:10:(length(sol.t)-1)
        lines3 = lines!(ax, x, 1e4.*sol.u[i][:, 1], color = (scale[k], 0.7),
        linewidth = minor_line)
        k+=1
    end
    lastline = lines!(ax, x, 1e4.*sol.u[end][:, 1], color = final_color, label = "Profile at $(round(pore_volume(sol.t[end]),digits=0)) PV",
        linewidth = major_line)
    xl = 0:0.01:L
    ss_line = cₐ .- rate_ss/v .* xl
    ss_line = collect(ss_line)
    ss_line[ss_line .< ca_min_from_c2max] .= ca_min_from_c2max
    # A_l = Fk .* (ss_line .+ Ka)./ss_line
    # cd_line = A_l .* Kd ./ (1 .- A_l)
    lines!(ax, xl, 1e4.*ss_line, color = crimson, linestyle = :dash, label = "Analytical prediction",
        linewidth = major_line)
    Label(fig[1:3, 1:5, Top()], halign = :left, L"\times 10^{-4}", fontsize = 10)
    text!(ax, L"C_A = C_A^{in} - r_{ss}/v \times x" ,position=(maximum(x)*0.2, 1e4.*(cₐ .- rate_ss/v .* maximum(x)*0.2)-1.),
        align = (:left, :bottom), rotation = atan(-1e4*rate_ss/v,1)-0.15, fontsize = line_text)
    Label(fig[1:3, 1:5, Top()], halign = :center, "a. Electron acceptor", fontsize = title)
    # axislegend(position = :rt)

    # Second graphh
    ax2 = Axis(fig[4:6, 1:5], xlabel = "x [m]", xlabelsize = labelsize,
        xticklabelsize = ticksize,
        ylabel = L"$C_D$ [mol L^{-1}]", ylabelsize = labelsize,
        yticklabelsize = ticksize,
        xgridwidth = 0.5, ygridwidth = 0.5,
        width = width, height = height)
    ax2.xticks = 0:2:10
    ax2.yticks = 0:1:3
    ylims!(ax2, -1e-2, 3.1)
    Kd_line = Kd .* ones(size(x,1))*1e5
    # cd_min_line = cd_min .* ones(size(x,1))*1e5
    lines = lines!(ax2, x, 1e5.*sol.u[1][:,3], color = (ashgray, 0.0), #label = "Transient profiles",
        linewidth = minor_line)
    k = 1
    for i in pvs_index1
        lines3 = lines!(ax2, x, 1e5.*sol.u[i][:, 3], color = (scale[k], 0.7),
        linewidth = 2.9)
        k+=1
    end
    lastline = lines!(ax2, x, 1e5.*sol.u[end][:, 3], color = final_color, label = "Profile at $(pore_volume(sol.t[end])) PV",
        linewidth = major_line)
    # lines!(ax2, xl, 1e5.*cd_line, color = crimson, linestyle = :dash, label = "Advection steady-state prediction",
    #     linewidth = major_line)
    Label(fig[4:6, 1:5, Top()], halign = :left, L"\times 10^{-5}", fontsize = 10)
    Label(fig[4:6, 1:5, Top()], halign = :center, "c. Electron donor", fontsize = title)
    # axislegend(position = :rt)

    ax3 = Axis(fig[1:3, 6:10], xlabelsize = labelsize,
        xticklabelsize = ticksize,
        ylabel = L"$r_{C_A}$ [mol L⁻¹ s⁻¹]", ylabelsize = labelsize,
        yticklabelsize = ticksize,
        xgridwidth = 0.5, ygridwidth = 0.5,
        width = width, height = height)
    ax3.xticks = 0:2:10
    ax3.yticks = 0:2:6.3
    ylims!(ax3, -1e-9, 7)
    ss_r_line = rate_ss .* ones(size(xl,1)).*1e10
    ss_r_line[ss_line .<= ca_min_from_c2max] .= 0
    bss_line2 = Bss2 .* ones(size(xl,1)).*1e4
    bss_line2[ss_line .<= ca_min_from_c2max] .= 0
    bss_line1 = Bss1 .* ones(size(xl,1)).*1e4
    bss_line1[ss_line .<= ca_min_from_c2max] .= 0
    Label(fig[1:3, 6:10, Top()], halign = :left, L"\times 10^{-10}", fontsize = 10)
    Label(fig[1:3, 6:10, Top()], halign = :center, "b. Reaction rate", fontsize = title)
    #text!(ax3, L"r_{ss} =\frac{k_{dec}}{Y_A} \left( \frac{\alpha_E}{k_{dec} (1/Y_D - \eta)} - K_B \right) " ,
    # position=(maximum(x)*0.0, rate_ss.*1e10+0.1),
    # align = (:left, :bottom), fontsize = line_text)
    Fk = sol.u[1][:,3]./(sol.u[1][:,3].+Kd) .* sol.u[1][:,1]./(sol.u[1][:,1].+Ka)
    lines = lines!(ax3, x, -1e10.*ca_rate_ferm(Fk, sol.u[1][:, 5], μ₂, Ya), color = (ashgray,0.7), label = "Time: $(sol.t[1] / 86400) days",
        linewidth = minor_line)
    k = 1
    for i in pvs_index1
        Fk = sol.u[i][:,3]./(sol.u[i][:,3].+Kd) .* sol.u[i][:,1]./(sol.u[i][:,1].+Ka)
        lines = lines!(ax3, x, -1e10.*ca_rate_ferm(Fk, sol.u[i][:, 5], μ₂, Ya), color = (scale[k],0.7),
        linewidth = minor_line)
        k+=1
    end
    Fk = sol.u[end][:,3]./(sol.u[end][:,3].+Kd) .* sol.u[end][:,1]./(sol.u[end][:,1].+Ka)
    last_line = lines!(ax3, x, -1e10.*ca_rate_ferm(Fk, sol.u[end][:, 5], μ₂, Ya),
     color = final_color, # label = "Transient Profiles",
    linewidth = major_line)
    lines!(ax3, xl, ss_r_line, color = crimson, linestyle = :dash, label = "steady-state reaction rate",
        linewidth = major_line)

    ax4 = Axis(fig[4:6,6:10], xlabel = "x [m]", xlabelsize = labelsize,
        xticklabelsize = ticksize,
        ylabel = "B [mol L⁻¹]", ylabelsize = labelsize,
        yticklabelsize = ticksize,
        xgridwidth = 0.5, ygridwidth = 0.5,
        width = width, height = height)
    ax4.xticks = 0:2:10
    ax4.yticks = 0:0.6:2
    ylims!(ax4, -1e-9, 2.2)
    # text!(ax4, L"B_{ss} = \frac{\alpha_E}{k_{dec}(1/Y_D - \eta)} - K_B" ,
    # position=(maximum(x)*0.03, Bss.*1e4+0.1),
    # align = (:left, :bottom), fontsize = line_text)
    
    lines = lines!(ax4, x, sol.u[1][:,4].*1e4, color = (ashgray,0.0), label = "Time: $(sol.t[1] / 86400) days",
        linewidth = minor_line)
    k = 1
    for i in pvs_index1
        lines = lines!(ax4, x, sol.u[i][:,4].*1e4, color = (scale[k],0.7), label = "Transient profiles",
        linewidth = minor_line)
        k+=1
    end
    last_line = lines!(ax4, x, sol.u[end][:,4].*1e4, color = final_color, label = "Final Profile",
    linewidth = major_line)

    lines!(ax4, xl, bss_line2, color = crimson, linestyle = :dash, label = "Steady-state biomass concentration",
        linewidth = major_line)

    Label(fig[4:6, 6:10, Top()], halign = :left, L"\times 10^{-4}", fontsize = 10)
    # add the scale bar
    Label(fig[4:6, 6:10, Top()], halign = :center, "d. Anaerobe Biomass", fontsize = title)

    # Second graphh
    ax5 = Axis(fig[7:9, 1:5], xlabel = "x [m]", xlabelsize = labelsize,
    xticklabelsize = ticksize,
    ylabel = L"$C_F$ [mol L^{-1}]", ylabelsize = labelsize,
    yticklabelsize = ticksize,
    xgridwidth = 0.5, ygridwidth = 0.5,
    width = width, height = height)
    ax5.xticks = 0:2:10
    ax5.yticks = 0:1:3
    ylims!(ax5, -1e-2, 3.1)
    Kd_line = Kd .* ones(size(x,1))*1e5
    # cd_min_line = cd_min .* ones(size(x,1))*1e5
    lines = lines!(ax5, x, 1e5.*sol.u[1][:,2], color = (ashgray, 0.0), #label = "Transient profiles",
        linewidth = minor_line)
    k = 1
    for i in pvs_index1
        lines3 = lines!(ax5, x, 1e5.*sol.u[i][:, 2], color = (scale[k], 0.7),
        linewidth = 2.9)
        k+=1
    end
    lastline = lines!(ax5, x, 1e5.*sol.u[end][:, 2], color = final_color, label = "Profile at $(pore_volume(sol.t[end])) PV",
        linewidth = major_line)
    # lines!(ax2, xl, 1e5.*cd_line, color = crimson, linestyle = :dash, label = "Advection steady-state prediction",
    #     linewidth = major_line)
    Label(fig[7:9, 1:5, Top()], halign = :left, L"\times 10^{-5}", fontsize = 10)
    Label(fig[7:9, 1:5, Top()], halign = :center, "e. Fermentation substrate", fontsize = title)
    # axislegend(position = :rt)

    ax6 = Axis(fig[7:9,6:10], xlabel = "x [m]", xlabelsize = labelsize,
        xticklabelsize = ticksize,
        ylabel = "B [mol L⁻¹]", ylabelsize = labelsize,
        yticklabelsize = ticksize,
        xgridwidth = 0.5, ygridwidth = 0.5,
        width = width, height = height)
    ax6.xticks = 0:2:10
    ax6.yticks = 0:0.6:2
    ylims!(ax6, -1e-9, 2.2)
    # text!(ax6, L"B_{ss} = \frac{\alpha_E}{k_{dec}(1/Y_D - \eta)} - K_B" ,
    # position=(maximum(x)*0.03, Bss.*1e4+0.1),
    # align = (:left, :bottom), fontsize = line_text)
    
    lines = lines!(ax6, x, sol.u[1][:,4].*1e4, color = (ashgray,0.0), label = "Time: $(sol.t[1] / 86400) days",
        linewidth = minor_line)
    k = 1
    for i in pvs_index1
        lines = lines!(ax6, x, sol.u[i][:,4].*1e4, color = (scale[k],0.7), label = "Transient profiles",
        linewidth = minor_line)
        k+=1
    end
    last_line = lines!(ax6, x, sol.u[end][:,4].*1e4, color = final_color, label = "Final Profile",
    linewidth = major_line)

    lines!(ax6, xl, bss_line1, color = crimson, linestyle = :dash, label = "Steady-state biomass concentration",
        linewidth = major_line)

    Label(fig[7:9, 6:10, Top()], halign = :left, L"\times 10^{-4}", fontsize = 10)
    # add the scale bar
    Label(fig[7:9, 6:10, Top()], halign = :center, "f. Fermenter Biomass", fontsize = title)



    fig[10,4:7] = Legend(fig, ax, framevisible = false, fontsize = 10, orientation = :horizontal, halign = :center, valign = :top)
    fig[11,3:8] = Colorbar(fig[11,3:8], limits = (minimum(pvs),maximum(pvs)),
     colormap=scale, label = "Pore Volumes", labelsize = 9, ticklabelsize = 8,
     vertical = false,
     height = 12, # width = Relative(0.5),
     ticks = round.(pvs,digits = 1))
    rowgap!(fig.layout, 2)
    colgap!(fig.layout, 3)
    # constrain the layout
    resize_to_layout!(fig)
    save(plotsdir("ca_cd_b_ferm.svg"), fig)
    save(plotsdir("ca_cd_b_ferm.png"), fig, px_per_unit = 1200/96)
end


# with_theme(theme_latexfonts()) do
#     fig = Figure(size = (504, 500))
#     ax = Axis(fig[1, 1:4], xlabel = "x [m]", xlabelsize = 15,
#         xticklabelsize = 14,
#         ylabel = L"$r_{C_A}$ [mol L⁻¹ s⁻¹]", ylabelsize = 15,
#         yticklabelsize = 14)
#     ax.xticks = 0:2:12
#     ylims!(ax, -1e-9, 6.5)
#     ss_line = rate_ss .* ones(size(x,1)).*1e10
#     bss_line = Bss1 .* ones(size(x,1)).*1e4
#     Label(fig[1, 1:4, Top()], halign = :left, L"\times 10^{-10}")
#     Label(fig[1, 1:4, Top()], halign = :right, "a.", fontsize = 20,
#     font = :bold)
#     lines!(ax, x, ss_line, color = :darkred, linestyle = :dash, label = "steady-state reaction rate",
#         linewidth = 2.5)
#     Fk = sol.u[1][:,3]./(sol.u[1][:,3].+Kd) .* sol.u[1][:,1]./(sol.u[1][:,1].+Ka)
#     lines = lines!(ax, x, -1e10.*ca_rate_ferm(Fk, sol.u[1][:, 5], μ₂, Ya), color = (scale[1],0.6), label = "Time: $(sol.t[1] / 86400) days",
#         linewidth = 2.9)
#     for i in 1:length(sol.t)
#         Fk = sol.u[i][:,3]./(sol.u[i][:,3].+Kd) .* sol.u[i][:,1]./(sol.u[i][:,1].+Ka)
#         lines = lines!(ax, x, -1e10.*ca_rate_ferm(Fk, sol.u[i][:, 5], μ₂, Ya), color = (scale[i],0.6), label = "Time: $(sol.t[i] / 86400) days",
#         linewidth = 2.9)
#     end
#     ax2 = Axis(fig[2,1:4], xlabel = "x [m]", xlabelsize = 15,
#         xticklabelsize = 14,
#         ylabel = "B [mol L⁻¹]", ylabelsize = 15,
#         yticklabelsize = 14)
#     ylims!(ax2, -1e-9, 2)
#     lines!(ax2, x, bss_line, color = :darkred, linestyle = :dash, label = "Steady-state biomass concentration",
#         linewidth = 2.5)
#     l = lines = lines!(ax2, x, sol.u[1][:,4].*1e4, color = (scale[1],0.6), label = "Time: $(sol.t[1] / 86400) days",
#         linewidth = 2.9)
#     l2 = lines!(ax2, x, sol.u[1][:,5].*1e4, color = (scale[1],0.6), label = "Time: $(sol.t[1] / 86400) days",
#         linewidth = 2.9)
#     for i in 1:length(sol.t)
#         lines = lines!(ax2, x, sol.u[i][:,4].*1e4, color = (scale[i],0.6), label = "Time: $(sol.t[i] / 86400) days",
#         linewidth = 2.9)
#         lines2 = lines!(ax2, x, sol.u[i][:,5].*1e4, color = (scale[i],0.6), label = "Time: $(sol.t[i] / 86400) days",
#         linewidth = 2.9)
#     end

#     Label(fig[2, 1:4, Top()], halign = :left, L"\times 10^{-4}")
#     # add the scale bar
#     Label(fig[2, 1:4, Top()], halign = :right, "b.", fontsize = 20,
#     font = :bold)
#     Colorbar(fig[1:2, 5], limits = (minimum(sol.t), maximum(sol.t))./86400, colormap=scale, label = "Time [days]", labelsize = 15, ticklabelsize = 14)
#     fig
#     save(plotsdir("reactBss_ferm.svg"), fig)
#     save(plotsdir("reactBss_ferm.png"), fig, px_per_unit = 1200/96)
# end