# based on: 1. https://discourse.julialang.org/t/deep-bayesian-model-discovery-without-using-neuralode-object/102403
# 2. https://docs.sciml.ai/Overview/stable/showcase/bayesian_neural_ode/
# 3. https://lux.csail.mit.edu/dev/tutorials/intermediate/2_BayesianNN

# Import libraries
using Lux, Turing, CairoMakie, Random, StableRNGs
using OrdinaryDiffEq, DifferentialEquations, ModelingToolkit, SciMLSensitivity
using ComponentArrays, Functors
using JLD2, LinearAlgebra

#========================================================================================================
    Experimental Settings
========================================================================================================#
experiment_series = "standard_nn"
chain_nr = ARGS[1] # array_nr is set to chain number
problem_name = ARGS[2] # problem_name = "seir"
experiment_name = ARGS[3] # experiment_name = "gaussian_IR_noise_0.01"
start_points_from_ensemble = "ensemble_1000/seir/multistart_gaussian_IR_noise_0.01"

# create paths and load experimental setting
experiment_path = joinpath(pwd(), "experiments/MCMC", experiment_series, problem_name, experiment_name)
include(joinpath(experiment_path, "experimental_setting.jl"))
isdir(joinpath(experiment_path, "log")) || mkpath(joinpath(experiment_path, "log"))
isdir(joinpath(experiment_path, "transitions")) || mkpath(joinpath(experiment_path, "transitions"))

#========================================================================================================
    Data creation
========================================================================================================#
include("../$problem_name.jl")
include("../simulate.jl")
# visualise_simulation(sample_data; model_id = 1)

#========================================================================================================
    UDE utils
========================================================================================================#
include("../utils.jl")

#========================================================================================================
    MCMC utils
========================================================================================================#
include("mcmc_utils.jl")
include("mcmc_$problem_name.jl")

# Perform inference.
nn_model, p_init, st = prepare_model(hyperpar, 1, sample_mechanistic, noise_model, true, component_vector=false)
ude_prob = ODEProblem(dynamics!, u0, tspan, p_init)
t_obs, X_obs, t_weights = sample_data(1)

# Define initial parameters
p_start = load(joinpath(pwd(), "experiments/ensemble/$start_points_from_ensemble", "models", "dataset_1_model_$(chain_nr)_p_opt.jld2"), "p_opt")[1:end]

rng = StableRNG(42);
model = bayes_nn(X_obs[[3,4],:], ude_prob)

## Using the iterator-interface ##
spl = DynamicPPL.Sampler(NUTS(0.65));

# Create an iterator we can just step through.
it = AbstractMCMC.Stepper(rng, model, spl, (nadapts=0,));
# Initial step
transition, state = AbstractMCMC.step(rng, model, spl; init_params=p_start)
# Simple container to hold the samples.
transitions = [transition];
# Simple condition that says we only want `num_samples` samples.
condition(spls) = spls < n_samples

current_samples = 1
# Sample until `condition` is no longer satisfied
while condition(current_samples)
    # For an iterator we pass in the previous `state` as the second argument
    transition, state = iterate(it, state)
    # Save `transition` if we're not adapting anymore
    push!(transitions, transition)

    if state.i % samples_per_subchain == 0
        println("Save at: ", state.i)
        save(joinpath(experiment_path, "transitions", "chain_$(chain_nr)_part_$(Int(state.i/samples_per_subchain)).jld2"), 
            Dict("transitions" => transitions, "state" => state))
        current_samples += length(transitions)
        transitions = []
    end
end

# """ Test loading the results

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

