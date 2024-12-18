#========================================================================================================
    UDE Definition
========================================================================================================#

function prepare_model(hyperpar, model_id, sample_mechanistic, noise_model, random_init; component_vector=true)
    θ_m_init = sample_mechanistic(model_id, random_init, noise_model)
    
    if random_init
        rng_nn = StableRNG(model_id)
    else
        rng_nn = StableRNG(42)
    end

    # combine parameters
    p_init = θ_m_init
    if component_vector # for MCMC we need the standard named tuple
        p_init = ComponentVector(p_init)
    end

    p_init
end

#========================================================================================================
    helper functions
========================================================================================================#

function gaussian_nll(pred, obs, np)
    σ = exp(np)
    # log(σ) + 0.5* ((pred-obs)/σ)²
    return sum(np .+ 0.5*((pred.-obs)./σ).^2)/prod(size(pred))
end

function negBin_nll(pred, obs, np)
    pred = max.([1e-5], pred) # ensure positivity of predictions
    dispersion = 1 + exp(np)
    p_s = 1/dispersion  # success probability p = mu/σ²
    r = pred*p_s / (1-p_s)  # number of successes
    return -sum(logpdf.(NegativeBinomial.(r, p_s), obs)) / prod(size(pred))
end

function mse(pred, obs, noise_par=missing)
    return mean((pred.-obs).^2)
end

if noise_model=="Gaussian"
    base_loss = gaussian_nll
elseif noise_model=="negBin"
    base_loss = negBin_nll
else
    base_loss = mse
end