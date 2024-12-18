# Create a regularization term and a Gaussian prior variance term.
sig = 3.0

# Specify the probabilistic model.
# Specify the probabilistic model.
@model function bayes_nn(data, ude_prob)

    # Sample the parameters
    nparameters = Lux.parameterlength(p_init.nn)
   
    par_α ~ Normal(0.0, 1.0) # transformed
    par_γ ~ Normal(0.0, 1.0) # transformed
    par_np ~  Normal(0.0, 1.0) # transformed
    par_nn ~ MvNormal(zeros(nparameters), sig .* ones(nparameters))
    
    parameters = merge((par_α = par_α, par_γ=par_γ), (nn=vector_to_parameters(par_nn, p_init.nn),))

    predicted = solve(ude_prob; p=parameters, saveat=t_obs, save_idxs=[3,4])

    pred = max.([1e-5], predicted) # ensure positivity of predictions
    dispersion = 1 + exp(par_np)
    p_s = 1/dispersion  # success probability p = mu/σ²
    r = pred*p_s / (1-p_s)  # number of successes

    for i in 1:length(predicted)
        data[:, i] ~ NegativeBinomial.(r[i], p_s) # par_np = log(σ²)
    end

    return nothing
end
