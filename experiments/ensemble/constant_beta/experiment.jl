using OrdinaryDiffEq, ModelingToolkit, SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using LinearAlgebra, Statistics
using DifferentialEquations # for automatic solver selection
using ComponentArrays, Lux, Zygote, Plots, StableRNGs, FileIO
using Distributions
gr()

#========================================================================================================
    Experimental Settings
========================================================================================================#
experiment_series = "constant_beta"
problem_name = ARGS[2] # problem_name = "seir"
experiment_name = ARGS[3] # experiment_name = "multistart_gaussian_IR_noise_0.01"

# create paths and load experimental setting
experiment_path = joinpath(pwd(), "experiments/ensemble", experiment_series, problem_name, experiment_name)
include(joinpath(experiment_path, "experimental_setting.jl"))
isdir(joinpath(experiment_path, "log")) || mkpath(joinpath(experiment_path, "log"))
isdir(joinpath(experiment_path, "models")) || mkpath(joinpath(experiment_path, "models"))

include("$problem_name.jl")
#========================================================================================================
    Data creation
========================================================================================================#
include("../../simulate.jl")

# visualise_simulation(sample_data; model_id = 1)

#========================================================================================================
    UDE utils
========================================================================================================#
include("utils.jl")
include("train.jl")

#========================================================================================================
    Training
========================================================================================================#

# predefine st
p_init = prepare_model(hyperpar, 1, sample_mechanistic, noise_model, random_init)

try
    global array_nr = parse(Int, ARGS[1])
catch 
    global array_nr = 1
end

models_per_cpu = Int(aleatoric_noise_realizations*(ensemble_members/2)/n_cpus) # set later back to: Int(aleatoric_noise_realizations*ensemble_members/n_cpus)
for (data_id, model_id) in collect(Iterators.product(1:aleatoric_noise_realizations, (ensemble_members/2+1):ensemble_members))[1+models_per_cpu*(array_nr-1):models_per_cpu*array_nr]
# Set later back to: for (data_id, model_id) in collect(Iterators.product(1:aleatoric_noise_realizations, 1:ensemble_members))[1+models_per_cpu*(array_nr-1):models_per_cpu*array_nr]
    data_id, model_id = Int(data_id), Int(model_id)
    try
        # println(model_id)
        t_obs, X_obs, t_weights = sample_data(data_id)
        t_train, X_train, t_val, X_val = train_validation_split(t_obs, X_obs, n_val, model_id)
        p_init = prepare_model(hyperpar, model_id, sample_mechanistic, noise_model, random_init)
        train_model(p_init, hyperpar, tspan, t_train, X_train, t_weights, t_val, X_val, joinpath(experiment_path, "models/"), model_id, data_id)
    catch e
        println(data_id, model_id)
        println(": $e")
    end
end

#=
model_id = 30
t_obs, X_obs, t_weights = sample_data(model_id)
t_train, X_train, t_val, X_val = train_validation_split(t_obs, X_obs, n_val, model_id)
seir_train_test_split(t_train, X_train, t_val, X_val)
quadratic_dynamics_train_test_split(t_train, X_train, t_val, X_val)
=#