using Lux, Turing, CairoMakie, Random, StableRNGs
using OrdinaryDiffEq, DifferentialEquations, ModelingToolkit, SciMLSensitivity
using ComponentArrays, Functors
using JLD2, LinearAlgebra
using Pigeons
using DynamicPPL
using DataFrames, CSV

experiment_series = "pigeon/seir"
experiment_name = "gaussian_IR_noise_0.01"

# create paths and load experimental setting
experiment_path = joinpath(pwd(), "experiments/MCMC", experiment_series, experiment_name)
include(joinpath(experiment_path, "experimental_setting.jl"))
# overwrite IC
include(joinpath(problem_name, replace(experiment_name, "multistart_" => ""), "experimental_setting.jl"))

problem_name = "seir"

result_path = joinpath(@__DIR__, problem_name, replace(experiment_name, "multistart_" => ""), "MCMC")

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
include("../MCMC/mcmc_utils.jl")
include("../MCMC/mcmc_$problem_name.jl")


#========================================================================================================
    Load & Evaluate results
========================================================================================================#
pt = PT(joinpath(experiment_path, "results", "latest"))

using MCMCChains
using StatsPlots

get_sample(pt)
sample_names(pt)

pt.shared.reports.summary

#========================================================================================================
    Create Plots (first: Density Plots)
========================================================================================================#
plot_path = joinpath(result_path, "plots")
isdir(joinpath(result_path, "plots", "traceplot")) || mkpath(joinpath(result_path, "plots", "traceplot"))

include("../evaluation_utils.jl")
include("../MCMC/eval_utils.jl")

mcmc_res = Chains(pt)

if noise_model=="Gaussian"
    x_name = "standard deviation"
elseif noise_model=="negBin"
    x_name = "overdispersion parameter"
end

# Create posterior summary files (posterior_stats.csv, posterior_trajectory.csv) based on MCMC samples
nn_model, p_init, st = prepare_model(hyperpar, 1, sample_mechanistic, noise_model, true, component_vector=false)
create_posterior_summary_files(result_path, mcmc_res, 8192, p_init) # as many as we used as base for ensemble based UQ

parameters = mcmc_res.value[:,1:end-1,:]
if occursin("variational", experiment_series)
    save(joinpath(result_path, "parameters.jld2"), "parameters", cat(parameters.data[:,1:end-1,[1]], parameters.data[:,1:end-1,[2]], dims=1), "names", sample_names(pt)[1:end-1])
else
    save(joinpath(result_path, "parameters.jld2"), "parameters", parameters.data[:,1:end-1,:], "names", sample_names(pt)[1:end-1])
end

#========================================================================================================
    Trace Plots and Density Plots
========================================================================================================#
pred = CSV.read(joinpath(result_path, "posterior_trajectory.csv"), DataFrame)
pred_agg = combine(groupby(pred, "t"),
    :S => minimum,
    :S => (q -> quantile(q, 0.005)) => :S_q_p5,
    :S => mean,
    :S => (q -> quantile(q, 0.995)) => :S_q_99p5,
    :S => maximum,
    :E => minimum,
    :E => (q -> quantile(q, 0.005)) => :E_q_p5,
    :E => mean,
    :E => (q -> quantile(q, 0.995)) => :E_q_99p5,    
    :E => maximum,
    :I => minimum,
    :I => (q -> quantile(q, 0.005)) => :I_q_p5,
    :I => mean,
    :I => (q -> quantile(q, 0.995)) => :I_q_99p5,    
    :I => maximum,
    :R => minimum,
    :R => (q -> quantile(q, 0.005)) => :R_q_p5,
    :R => mean,
    :R => (q -> quantile(q, 0.995)) => :R_q_99p5,    
    :R => maximum,
    :β => minimum,
    :β => (q -> quantile(q, 0.005)) => :β_q_p5,
    :β => mean,
    :β => (q -> quantile(q, 0.995)) => :β_q_99p5,    
    :β => maximum)
CSV.write(joinpath(result_path, "aggregate_predictions.csv"), pred_agg)
fig = plot_summary_overview(pred_agg)
save(joinpath(plot_path, "trajectory_summary.png"), fig)

