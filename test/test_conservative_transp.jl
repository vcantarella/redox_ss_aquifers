using DrWatson
@quickactivate "redox_steady_state_aquifers"
using DifferentialEquations
using Test
using SpecialFunctions
using CairoMakie
using Symbolics
"""
constant_injection(cr, x, t, c0, c_in, v, Dl)

Calculates the concentration profile of a solute undergoing
advection-dispersion transport in a porous 1D domain with constant
 in a 1D domain with constant injection
Original reference: (Ogata & Banks, 1952): 
Appelo, C.A.J.; Postma, Dieke. Geochemistry, Groundwater and Pollution (p. 104).

# Arguments
- `cr::Matrix`: A 2D array to store the concentration profile of a solute. 
    first dimension is time and second dimension is space.
- `x::Vector`: A 1D array of spatial locations. (Where to sample the concentration)
- `t::Vector`: A 1D array of time locations. (When to sample the concentration)
- `c0::Real`: The concentratio at x=0 (inflow concentration).
- `c_in::Real`: The initial concentration in the column (t=0).
- `v::Real`: The velocity of the solute.
- `Dl::Real`: The longitudinal dispersion coefficient.
# Returns
    nothing, the results are stored in the `cr` array. 
"""
function constant_injection(
     cr::Matrix,
     x::Vector,
     t::Vector,
     c0::Real,
     c_in::Real,
     v::Real,
     Dl::Real,
     )
    
    for i in eachindex(x)
        cr[:, i] .= c_in .+ (c0 - c_in) / 2 .* erfc.((x[i] .- v .* t)
         ./ (2 .* sqrt.(Dl .* t)))
    end
    return nothing
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

include(srcdir("ode_model.jl"))
fixed_decay!, ca_rate_fd= create_fixed_decay(v, D, dx, c_in, nmob)


u0
p
# Making the parameters 0:
p[1] = 0.
p[3] = 0.
p[8] = 0.

# Running the model:
jac_sparsity = Symbolics.jacobian_sparsity((du, u) -> fixed_decay!(du, u, p, 1),
    du0, u0) # add the sparsity pattern to speed up the solution
const_rhs! = ODEFunction(fixed_decay!, jac_prototype=jac_sparsity)
prob = ODEProblem(const_rhs!, u0, tspan, p)
# solving the problem
sol = solve(prob, Tsit5(), reltol = 1e-8, abstol = 1e-8)
sol.t
arr = zeros(length(sol.t), size(u0, 1))
cr = zeros(length(sol.t), size(u0, 1))
# Check discrepance between the analytical and numerical solutions
norm_mean_squared_error(a, b) = sqrt(sum((a-b).^2)/length(a))/(maximum(a)-minimum(a))

# Checking for each mobile species the fit to the analytical solution:
for j in 1:nmob
    constant_injection(cr, collect(x), sol.t, c_in[j], u0[j], v, D[j])
    for i in eachindex(sol.t)
        arr[i,:] = sol.u[i][:,1]'
    end
    @test norm_mean_squared_error(cr[2:end,:], arr[2:end,:]) < 5e-2
end

# Plotting the results
fig = Figure(backgroundcolor = RGBf(0.98, 0.98, 0.98), size = (500, 800))
axis = []
for j in 1:nmob
    ax = Axis(fig[j, 1], xlabel = "x [m]", xlabelsize = 12,
        xticklabelsize = 11,
        ylabel = L"$concentration$ ", ylabelsize = 12,
        yticklabelsize = 11)
    push!(axis, ax)
    #ax.xticks = 0:2:10
    #ax.yticks = 0:1e-4:6e-4
    #ylims!(ax, -1e-9, 6.3e-4)
    constant_injection(cr, collect(x), sol.t, c_in[j], u0[j], v, D[j])
    for i in 1:50:length(sol.t)
        lines = lines!(axis[j], x, sol.u[i][:, j], color = :blue)
        points = scatter!(axis[j], x[1:100:length(x)], cr[i, 1:100:length(x)], color = :red)
    end
end
fig