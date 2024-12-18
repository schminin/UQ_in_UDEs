
function Pigeons.initialization(target::ModelType, rng::AbstractRNG, ::Int64) 
    result = DynamicPPL.VarInfo(rng, target.model, DynamicPPL.SampleFromPrior(), DynamicPPL.PriorContext())
    DynamicPPL.link!!(result, DynamicPPL.SampleFromPrior(), target.model)

    # custom init goes here: 
    fieldnames(typeof(result.metadata))
    Pigeons.update_state!(result, :par_α, 1, p_start.par_α)
    Pigeons.update_state!(result, :par_np, 1, log(exp(p_start.np)^2)) # for ensemble np is log(σ), for mcmc log(σ²)
    for i in 1:length(p_start.nn)
        Pigeons.update_state!(result, :par_nn, i, p_start.nn[i])
    end

    return result
end