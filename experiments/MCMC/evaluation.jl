# Import libraries
using Lux, Turing, CairoMakie, Random, StableRNGs
using OrdinaryDiffEq, DifferentialEquations, ModelingToolkit, SciMLSensitivity
using ComponentArrays, Functors
using JLD2
using LinearAlgebra, DataFrames, CSV

using StatsPlots
using Plots
# using MCMCChains

#========================================================================================================
    Experimental Settings
========================================================================================================#
experiment_series = "standard_nn"
problem_name = "seir" # ARGS[2]
experiment_name = "gaussian_IR_noise_0.01" # ARGS[3] 

# create paths and load experimental setting
experiment_path = joinpath(pwd(), "experiments/MCMC", experiment_series, problem_name, experiment_name)
include(joinpath(experiment_path, "experimental_setting.jl"))
include("../$problem_name.jl")
include("eval_utils.jl")
include("../evaluation_utils.jl")
# ude utils
include("../utils.jl")
# to access training data
include("../simulate.jl")
include("mcmc_utils.jl")
include("mcmc_$problem_name.jl")

nn_model, p_init, st = prepare_model(hyperpar, 1, sample_mechanistic, noise_model, true, component_vector=false)
ude_prob = ODEProblem(dynamics!, u0, tspan, p_init)
t_obs, X_obs, t_weights = sample_data(1)
model = bayes_nn(X_obs[[3,4],:], ude_prob)
spl = DynamicPPL.Sampler(NUTS(0.65));

using Plots
using StatsPlots

plot_path = joinpath(experiment_path, "plots")
isdir(joinpath(experiment_path, "plots", "density")) || mkpath(joinpath(experiment_path, "plots", "density"))

#========================================================================================================
    Posterior Evaluation
========================================================================================================#

function load_chain(chain_nr, experiment_path)
    t = []

    for i in 1:100
        t = vcat(t, load(joinpath(experiment_path, "transitions", "chain_$(chain_nr)_part_$i.jld2"), "transitions"))
    end
    s = load(joinpath(experiment_path, "transitions", "chain_$(chain_nr)_part_100.jld2"), "state")
    # Finally, if you want to convert the vector of `transitions` into a
    # `MCMCChains.Chains` like is typically done:
    chain = AbstractMCMC.bundle_samples(
        map(identity, t),  # trick to concretize the eltype of `transitions`
        model,
        spl,
        s,
        MCMCChains.Chains
    )
    return chain
end

chain_1 = load_chain(1, experiment_path)
chain_2 = load_chain(2, experiment_path)
chain_3 = load_chain(3, experiment_path)
chain_4 = load_chain(4, experiment_path)
chain_5 = load_chain(5, experiment_path)
chain_6 = load_chain(6, experiment_path)
chain_7 = load_chain(7, experiment_path)
chains = cat(chain_1, chain_2, chain_3, chain_4, chain_5, chain_6, chain_7, dims=3)
parameters = chains.value[:, 1:length(ComponentVector(p_init)[1:end]), :]

create_posterior_summary_files(experiment_path, chains, 10000, p_init) # as many as we used as base for ensemble based UQ

#========================================================================================================
    Plots
========================================================================================================#
trace_plot("alpha", parameters, 1, retrieve_α, 1:size(parameters)[1], 1:7)
trace_plot("gamma", parameters, 2, retrieve_γ, 1:size(parameters)[1], 1:7)
trace_plot("σ", parameters, 3, x -> sqrt(exp(x)), 1:size(parameters)[1], 1:7)
trace_plot("nn_1", parameters, 4, identity, 1:size(parameters)[1], 1:7)


# Evaluation specific for SEIR
# density of mechanistic parameters
density_with_reference(p_ode[1], "α", retrieve_α.(vcat(Array(get(chains, :par_α).par_α)...)), :steelblue4; save_plot=true, plot_path=joinpath(plot_path, "density"))
density_with_reference(p_ode[2], "γ", retrieve_γ.(vcat(Array(get(chains, :par_γ).par_γ)...)), :darkslategray4; save_plot=true, plot_path=joinpath(plot_path, "density"))
# Distribution of the np
if noise_model=="Gaussian"
    x_name = "standard deviation"
elseif noise_model=="negBin"
    x_name = "overdispersion parameter"
end
density_with_reference(noise_par, x_name, sqrt.(exp.(vcat(Array(get(chains, :par_np).par_np)...))), :lightblue; save_plot=true, plot_path=joinpath(plot_path, "density"))

fig = constant_parameter_summary(chains)
save(joinpath(plot_path, "density", "summary.png"), fig)

pred_agg = aggregate_predictions(CSV.read(joinpath(experiment_path, "posterior_trajectory.csv"), DataFrame), 1, problem_name)
CSV.write(joinpath(experiment_path, "aggregated_predictions.csv"), pred_agg)

fig = plot_summary_overview(pred_agg)
save(joinpath(plot_path, "trajectory_summary.png"), fig)
