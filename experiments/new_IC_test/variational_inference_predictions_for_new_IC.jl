# Import libraries
using Lux, Turing, CairoMakie, Random, StableRNGs
using Turing: Variational
using OrdinaryDiffEq, DifferentialEquations, ModelingToolkit, SciMLSensitivity
using ComponentArrays, Functors
using JLD2, LinearAlgebra
using DataFrames, CSV
#========================================================================================================
    Experimental Settings
========================================================================================================#
experiment_series = "standard_nn/"
problem_name = "seir" # problem_name = ARGS[2] # 
experiment_name = "gaussian_IR_noise_0.01" # experiment_name = ARGS[3] # 

# create paths and load experimental setting
experiment_path = joinpath(pwd(), "experiments/variational_inference", experiment_series, problem_name, experiment_name)
include(joinpath(experiment_path, "experimental_setting.jl"))
# overwrite IC
include(joinpath(problem_name, replace(experiment_name, "multistart_" => ""), "experimental_setting.jl"))
result_path = joinpath(@__DIR__, problem_name, replace(experiment_name, "multistart_" => ""), "variational_inference")

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
    VI utils
========================================================================================================#
include("../variational_inference/vi_$problem_name.jl")
include("../variational_inference/vi_utils.jl")
include("../MCMC/eval_utils.jl")
include("../evaluation_utils.jl")

nn_model, p_init, st = prepare_model(hyperpar, 1, sample_mechanistic, noise_model, true, component_vector=false)
ude_prob = ODEProblem(dynamics!, u0, tspan, p_init)

q = load(joinpath(experiment_path, "posteriors", "advi_100000.jld2"))["q"]
sampled_data = rand(q, 10000)

u0

df_stats = DataFrame()
df_trajectory = DataFrame()
for (sample_id, p) in enumerate(eachcol(Array(sampled_data)))
    par = merge((par_α = p[1], par_γ=p[2], np=sqrt.(exp.(p[3]))), (nn=vector_to_parameters(p[4:end], p_init.nn),))
    df_stats_sub, df_trajectory_sub = evaluate_parameter_sample(sample_id, par)
    append!(df_stats, df_stats_sub, cols=:union)
    append!(df_trajectory, df_trajectory_sub, cols=:union)
end
CSV.write(joinpath(result_path, "posterior_stats.csv"), df_stats)
CSV.write(joinpath(result_path, "posterior_trajectory.csv"), df_trajectory)

pred_agg = aggregate_predictions(CSV.read(joinpath(result_path, "posterior_trajectory.csv"), DataFrame), 1, problem_name)
CSV.write(joinpath(result_path, "aggregated_predictions.csv"), pred_agg)
