using DifferentialEquations
using Test
using SpecialFunctions
using CairoMakie
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
include("../scripts/steady_state_demo.jl")

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
sol = solve(prob, Rosenbrock23(), reltol = 1e-9, abstol = 1e-9)
sol.t
arr = zeros(length(sol.t), size(u0, 1))
cr = zeros(length(sol.t), size(u0, 1))
# Checking for each mobile species the fit to the analytical solution:
for j in 1:nmob
    constant_injection(cr, collect(x), sol.t, c_in[j], u0[j], v, D[j])
    for i in eachindex(sol.t)
        arr[i,:] = sol.u[i][:,1]'
    end
    @test isapprox(cr[2:end,:], arr[2:end,:] , atol = u0[j]*0.1, rtol=5e-2)
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