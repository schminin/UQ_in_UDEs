# Import libraries
using Lux, Turing, CairoMakie, Random, StableRNGs
using OrdinaryDiffEq, DifferentialEquations, ModelingToolkit, SciMLSensitivity
using ComponentArrays, Functors
using JLD2, LinearAlgebra
using Pigeons
using DynamicPPL

println("number of threads: $(Threads.nthreads())")

#========================================================================================================
    Experimental Settings
========================================================================================================#
experiment_series = "pigeon"
problem_name = ARGS[1] # problem_name = "quadratic_dynamics"
experiment_name = ARGS[2] # experiment_name = "gaussian_noise_0.05"

# create paths and load experimental setting
experiment_path = joinpath(pwd(), "experiments/MCMC", experiment_series, problem_name, experiment_name)
include(joinpath(experiment_path, "experimental_setting.jl"))
isdir(joinpath(experiment_path, "log")) || mkpath(joinpath(experiment_path, "log"))
isdir(joinpath(experiment_path, "results")) || mkpath(joinpath(experiment_path, "results"))

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
if occursin("seir", problem_name)
    if noise_model=="negBin"
        include("mcmc_seir_negbin.jl")
    else
        include("mcmc_seir.jl")
    end
else
    include("mcmc_$problem_name.jl")
end

# Perform inference.
nn_model, p_init, st = prepare_model(hyperpar, 1, sample_mechanistic, noise_model, true, component_vector=false)
ude_prob = ODEProblem(dynamics!, u0, tspan, p_init)
t_obs, X_obs, t_weights = sample_data(1)

rng = StableRNG(42);
model = bayes_nn(X_obs[[1],:], ude_prob)

function define_target()
    return Pigeons.TuringLogPotential(model)
end

const ModelType = typeof(define_target())

include("gaussian_initialization_quadratic_dynamics.jl")

p_start = load(joinpath(pwd(), "experiments/ensemble/ensemble_10000", problem_name, "multistart_$experiment_name","models", "model_5_p_opt.jld2"), "p_opt")

include(joinpath(experiment_path, "pigeons_setting.jl"))

pt = pigeons(pigeons_setting)
@assert Pigeons.variable(pt.replicas[1].state, :par_α) == [p_start.par_α]

println("Results stored in: $(pt.exec_folder)")
