#========================================================================================================
    Data simulation
========================================================================================================#
saveat_plot = 0.1
state_names = ["x"]
time_dependent_params = []

"""
model with quadratic dynamics 
"""
function quadratic_dynamics!(du, u, p, t)
    α, β = p
    x = u[1]
    du[1] = α*x - β*x^2
end

simulation_dynamics! = quadratic_dynamics!


function sample_mechanistic(sample_id, random_init, noise_model)
    rng_mech = StableRNG(sample_id)
    if random_init
        # sample from log uniform distribution
        if noise_model == "Gaussian"    # in parametrized space
            θ_m_init = (par_α = log(rand(rng_mech, distribution_α)), 
                        np = log(rand(rng_mech, distribution_np)) # log space ensures positivity
                        )
        else
            print("Noise Model not implemented")
        end
    else
        θ_m_init = (par_α = 0.0, 
                    np = log(0.05))
    end
    return θ_m_init
end

#========================================================================================================
    Reparametrization of mechanistic parameters
========================================================================================================#
function retrieve_α(α_parameterized)
    exp(α_parameterized)
end

function parametrize_α(α)
    log(α)
end

#========================================================================================================
    UDE
========================================================================================================#
# Define UDE
function ude_dynamics!(du, u, p, st, t)
    α = retrieve_α(p.par_α)
    subtrahend = nn_model([t], p.nn, st)[1][1] # βx²
    x = u[1]
    du[1] = α*x - subtrahend
end

dynamics!(du, u, p, t) = ude_dynamics!(du, u, p, st, t)
