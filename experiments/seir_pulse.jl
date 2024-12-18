#========================================================================================================
    Data simulation
========================================================================================================#
saveat_plot = 1.0
state_names = ["S", "E", "I", "R"]
time_dependent_params = ["β"]

"""
time dependent transmission rate
"""
function β(t)
    if 15<t<30
        β = 0.05
    else
        β = 0.5
    end
end

#=
t = 0:saveat_plot:130
plot(t, β.(t), label=missing, xlabel = "t", ylabel = "β")
savefig("model/time_dependent_beta/beta.png")
=#

"""
SEIR model 
"""
function seir!(du, u, p, t)
    α, γ, N = p
    S, E, I, R = u
    du[1] = -β(t)*S*I/N
    du[2] = β(t)*S*I/N - α*E
    du[3] = α*E - γ*I
    du[4] = γ*I
end

simulation_dynamics! = seir!

#=
p_ode = [0.9, 0.1, 1.] # α, γ, N 
simulation_prob = ODEProblem(simulation_dynamics!, u0, tspan, p_ode)
t = 0:saveat_plot:130
X = Array(solve(simulation_prob, saveat=t))
plot(t, X', label = ["S" "E" "I" "R"], xlabel = "t", ylabel = "count")
savefig("model/time_dependent_beta/beta.png")
=#

function sample_mechanistic(sample_id, random_init, noise_model)
    rng_mech = StableRNG(sample_id)
    if random_init
        # sample from log uniform distribution
        if noise_model == "Gaussian"    # in parametrized space
            θ_m_init = (par_α = rand(rng_mech, Normal(0,1)), 
                        par_γ = rand(rng_mech, Normal(0,1)),
                        np = log(rand(rng_mech, LogUniform(1e-3,1.0))) # log space ensures positivity
                        )
        else    # in parametrized space
            θ_m_init = (par_α = rand(rng_mech, Normal(0,1)), 
                        par_γ = rand(rng_mech, Normal(0,1)),
                        np = log(rand(rng_mech, Beta(2,2))) # noise paramater -> overdispersion = 1 + exp(np)
                        )
        end
    else
        θ_m_init = (par_α = 0.0, 
                    par_γ = 0.0,
                    np = log(10.0))
    end
    return θ_m_init
end

#========================================================================================================
    Reparametrization of mechanistic parameters
========================================================================================================#

"""
    based on a parametrized version of α, 
    calculate the corresponding α

    this parametrization ensures that 
        1.  0<α<24
        2.  values close to zero (corresponding to more stable NNs)
            are related to the more likely value region of α
    
    Note
    the latent period (inverse of alpha) could reasonably be anywhere from an hour 
    (e.g., for certain foodborne illnesses) to several years (e.g., certain malaria cases)
"""
function retrieve_α(α_parameterized)
    (tanh(α_parameterized-2) + 1)*12
end

function parametrize_α(α)
    atanh(α/12-1) + 2
end

"""
    based on a parametrized version of γ, 
    calculate the corresponding γ

    this parametrization ensures that 
        1.  0 < γ < 1
        2.  values close to zero (corresponding to more stable NNs)
            are related to the more likely value region of gamma
    Note
    we assume, that someone is at least infectious for 1 day
"""
function retrieve_γ(γ_parameterized)
    (tanh(γ_parameterized-1.5) + 1)*0.5
end

function parametrize_γ(γ)
    atanh(γ/1.5-1) + 0.5
end

"""
    based on a parametrized version of β, 
    calculate the corresponding β

    this parametrization ensures that 
        1.  0<β<3
        2.  values close to zero (corresponding to more stable NNs)
            are related to the more likely value region of beta (~0.05 to ~2.4)
    
    Note
    for beta, consider measles as the classic example of a highly transmissible disease
    (R_0 between 12 and 18 and duration of infectiousness of around a week implies beta between 1.7 and 2.6 per day)
"""
function retrieve_β(β_parameterized)
    (tanh(β_parameterized-1.5) + 1)*1.5
end

function parametrize_β(β)
    atanh(β/1.5-1) + 1.5
end

#========================================================================================================
    UDE
========================================================================================================#

# Define UDE
function ude_dynamics!(du, u, p, st, t; N=p_ode[3])
    β = retrieve_β(nn_model([t], p.nn, st)[1][1])
    S, E, I, R = u
    γ = retrieve_γ(p.par_γ)
    α = retrieve_α(p.par_α)
    du[1] = -β*S*I/N
    du[2] = β*S*I/N - α*E
    du[3] = α*E - γ*I
    du[4] = γ*I
end

dynamics!(du, u, p, t) = ude_dynamics!(du, u, p, st, t)
