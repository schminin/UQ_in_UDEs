simulation_prob = ODEProblem(simulation_dynamics!, u0, tspan, p_ode)

#========================================================================================================
    Helper Functions
========================================================================================================#
"""
Add negative binomial noise to observations
    mean:       mean value of negative binomial (i.e. output of model)
    dispersion: variance/mean ratio of negative binomial distribution 
"""
function add_negative_binomial_noise(mean, dispersion, rng)
    p_s = 1/dispersion  # success probability p = mu/σ²
    r = (mean + 1e-5) * p_s / (1-p_s) # number of successes
    return rand(rng, NegativeBinomial(r, p_s), 1)[1]
end


"""
Add Gaussian noise to observations
    mean:       mean value of Gaussian distribution (i.e. output of model)
    std:        standard deviation of Gaussian distribution
"""
function add_gaussian_noise(mean, std, rng)
    sample = rand(rng, Normal(mean, std), 1)[1]
    return max.(0.0, sample)
end


"""
Simulate based on the SEIR model
    n_timepoints:       number of timepoints to use as observed values
    noise_model:        one of "negBin" and "Gaussian"
    noise_par:          standard deviation in case of Gaussian noise, dispersion in case of negBin noise 
    fraction:           what is the fraction of timepoints we will observe for one ensemble method?
                        for bootstrapping, set e.g. to 0.8, else, set to 1
"""
function simulate_observations(n_timepoints, noise_model, noise_par, fraction=1, noise_seed=1; ode_prob)
    # create reference
    sol = solve(ode_prob, Tsit5(), abstol = 1e-12, reltol = 1e-12, saveat = 0.001)
    t = sol.t
    X = Array(sol)

    step_size = length(t)/(n_timepoints/fraction)
    idx = round.(Int, 1:step_size:length(t))
    t_obs = t[idx]
    X_obs = X[:, idx]

    rng_data = StableRNG(noise_seed)
    if noise_model=="Gaussian"
        X_obs = add_gaussian_noise.(X_obs, noise_par, rng_data)
    elseif noise_model=="negBin"
        X_obs = add_negative_binomial_noise.(X_obs, noise_par, rng_data)
    end

    return t_obs, X_obs
end

"""
create training sample for one model of non-parametric bootstrapping 
"""
function sample_bootstrap(t_obs, X_obs, n_timepoints, model_id)
        # based on a fixed dataset, sample 80% of the available timepoints (every timepoint may appear several times)
        rng_data = StableRNG(model_id)
        idx = sort(vcat(1, sample(rng_data, 2:length(t_obs), n_timepoints-1; replace=true)))
        t_train = t_obs[unique(idx)]
        X_train = X_obs[:, unique(idx)]
        t_weights = [count(i->(i==k), idx) for k in unique(idx)]
        return t_train, X_train, t_weights
    return t_train, X_train, t_weights
end

"""
create training sample for one model of repeated-holdout 
"""
function sample_repeated_holdout(t_obs, X_obs, n_timepoints, model_id)
    # based on a fixed dataset, sample 80% of the available timepoints (every timepoint only once)
    rng_data = StableRNG(model_id)
    idx = sort(vcat(1, sample(rng_data, 2:length(t_obs), n_timepoints-1; replace=false)))
    t_train = t_obs[idx]
    X_train = X_obs[:, idx]
    t_weights = repeat([1.0], length(t_train))
    return t_train, X_train, t_weights
end

"""
create training sample based on a resimulation of the model
"""
function resimulate(n_timepoints, model_id, noise_model, noise_par; ode_prob, tspan)
    rng_data = StableRNG(model_id)

    # resample timepoints with replacement for every model
    t_train = vcat(0.0, sort(rand(rng_data, n_timepoints-1) .* (tspan[end]-tspan[begin]) .+ tspan[begin]))
    X_train = Array(solve(ode_prob, Tsit5(), abstol = 1e-12, reltol = 1e-12, saveat = t_train))
    if length(unique(t_train))==n_timepoints # this should always be the case
        t_weights = repeat([1.0], n_timepoints)
    end

    rng_data = StableRNG(model_id)
    if noise_model=="Gaussian"
        X_train = add_gaussian_noise.(X_train, noise_par, rng_data)
    elseif noise_model=="negBin"
        X_train = add_negative_binomial_noise.(X_train, noise_par, rng_data)
    end

    return t_train, X_train, t_weights
end

"""
out-of-time sampling of n_val validation points
"""
function train_validation_split(t_obs, X_obs, n_val, model_id)
    rng_split = StableRNG(model_id)
    # out of time sampling of validation points (without IC)
    idx_val = sort(sample(rng_split, 2:length(t_obs), n_val; replace=false))
    # rest is training data
    idx_train = [id for id in 1:n_timepoints if !(id in idx_val)]
    # subselection of respective observations
    t_val = t_obs[vcat(1,idx_val)]
    X_val = X_obs[:,vcat(1,idx_val)]
    X_train = X_obs[:,idx_train]
    t_train = t_obs[idx_train]
    return t_train, X_train, t_val, X_val
end


function resample_noise(n_timepoints, noise_model, noise_par, subsample_faction, model_id; ode_prob)
    t_sim, X_sim = simulate_observations(n_timepoints, noise_model, noise_par, subsample_faction, model_id; ode_prob = ode_prob)
    return (t_sim, X_sim, repeat([1.0], length(t_sim)))
end

#========================================================================================================
    Apply to current settings
========================================================================================================#

if subsampling_method in ["quasi_bootstrap", "repeated_holdout"]
    t_sim, X_sim = simulate_observations(n_timepoints, noise_model, noise_par, subsample_fraction, 1; ode_prob = simulation_prob)
elseif subsampling_method=="none"
    t_sim, X_sim = simulate_observations(n_timepoints, noise_model, noise_par, 1, 1; ode_prob = simulation_prob)  
end

if subsampling_method=="bootstrap"
    sample_data = data_id -> sample_bootstrap(t_sim, X_sim, n_timepoints, data_id)
elseif subsampling_method=="repeated_holdout"
    sample_data = data_id -> sample_repeated_holdout(t_sim, X_sim, n_timepoints, data_id)
elseif subsampling_method=="resimulate"
    sample_data = data_id -> resimulate(n_timepoints, data_id, noise_model, noise_par; ode_prob = simulation_prob, tspan = tspan)
elseif subsampling_method=="resample_noise"
    sample_data = data_id -> resample_noise(n_timepoints, noise_model, noise_par, 1, data_id; ode_prob=simulation_prob)
else
    sample_data = data_id -> (t_sim, X_sim, repeat([1.0], length(t_sim)))
end
