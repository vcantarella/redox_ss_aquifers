
"""
    create_fixed_decay(v, De, dx, c_in, nmob)

Creates an ODE right-hand side (RHS) function to solve the coupled reactive-transport problem for the model formulation.

# Arguments
- `v::Number`: The velocity.
- `De::Vector`: The dispersion row vector.
- `dx::Number`: The grid spacing.
- `c_in::Vector`: The inflow concentration.
- `nmob::Int`: The number of mobile species.

# Returns
- A function representing the ODE RHS for the coupled reactive-transport problem.
- A function representing the Monod reaction rate for the electron acceptor
# Example
```julia
rhs, ca_rate = create_fixed_decay(1.0, [0.1, 0.2], 0.01, 1.0, 2)
```
"""
function create_fixed_decay(v, De, dx, c_in, nmob)
    # Defining the reaction rates model
    """
    monod_rate(Ca, Cd , B, μₘ, Ka, Kd, Ya)

    The Monod reaction rate for the electron acceptor in the paper.
    parameters are:
    - Ca: the concentration of the electron acceptor
    - Cd: the concentration of the electron donor
    - B: the concentration of the biomass
    - μₘ: the maximum specific growth rate
    - Ka: the half-saturation constant for the electron acceptor
    - Kd: the half-saturation constant for the electron donor
    - Ya: the yield coefficient for the electron acceptor

    """
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

    """
    fixed_decay!(du, u, p ,t)

    The ODE RHS function for the coupled reactive-transport problem in the paper written according to the DifferentialEquations convention.
    parameters are stored in the p vector.
    they are:
    - α: The hydrolysis rate.
    - Kb: the half-saturation constant for the biomass in the hydrolysis reaction
    - μₘ: the maximum specific growth rate
    - Ka: the half-saturation constant for the electron acceptor
    - Kd: the half-saturation constant for the electron donor
    - Ya: the yield coefficient for the electron acceptor
    - Yd: the yield coefficient for the electron donor
    - k_dec: the decay rate of the biomass
    - Ks: the half-saturation constant for the solid-phase electron donor in the hydrolysis reaction.
    - η: the fraction of the biomass that is converted to the electron donor in the decay reaction.

    """
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


