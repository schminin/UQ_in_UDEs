using Lux, Turing, CairoMakie, Random, StableRNGs
using OrdinaryDiffEq, DifferentialEquations, ModelingToolkit, SciMLSensitivity
using ComponentArrays, Functors
using JLD2, LinearAlgebra
using Pigeons
using DynamicPPL
using DataFrames, CSV

experiment_series = "pigeon/seir_pulse"
experiment_name = "gaussian_IR_noise_0.03"

# create paths and load experimental setting
experiment_path = joinpath(pwd(), "experiments/MCMC", experiment_series, experiment_name)
include(joinpath(experiment_path, "experimental_setting.jl"))

problem_name = "seir_pulse"
#========================================================================================================
    Data creation
========================================================================================================#
include("../../$problem_name.jl")
include("../../simulate.jl")
# visualise_simulation(sample_data; model_id = 1)

#========================================================================================================
    UDE utils
========================================================================================================#
include("../../utils.jl")

#========================================================================================================
    MCMC utils
========================================================================================================#
include("../mcmc_utils.jl")
if occursin("seir", problem_name)
    include("../mcmc_seir.jl")
else
    include("../mcmc_quadratic_dynamics.jl")
end


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
plot_path = joinpath(experiment_path, "plots")
isdir(joinpath(experiment_path, "plots", "distribution")) || mkpath(joinpath(experiment_path, "plots", "distribution"))
isdir(joinpath(experiment_path, "plots", "traceplot")) || mkpath(joinpath(experiment_path, "plots", "traceplot"))

include("../../evaluation_utils.jl")
include("../eval_utils.jl")

mcmc_res = Chains(pt)

if noise_model=="Gaussian"
    x_name = "standard deviation"
elseif noise_model=="negBin"
    x_name = "overdispersion parameter"
end

density_with_reference(p_ode[1], "α", retrieve_α.(vcat(Array(get(mcmc_res, :par_α).par_α)...)), :steelblue4; save_plot=true, plot_path=joinpath(plot_path, "distribution"))
density_with_reference(p_ode[2], "γ", retrieve_γ.(vcat(Array(get(mcmc_res, :par_γ).par_γ)...)), :darkslategray4; save_plot=true, plot_path=joinpath(plot_path, "distribution"))
# Distribution of the np
if noise_model=="Gaussian"
    density_with_reference(noise_par, x_name, sqrt.(exp.(vcat(Array(get(mcmc_res, :par_np).par_np)...))), :lightblue; save_plot=true, plot_path=joinpath(plot_path, "distribution"))
else
    density_with_reference(noise_par, x_name, 1 .+ exp.((vcat(Array(get(mcmc_res, :par_np).par_np)...))), :lightblue; save_plot=true, plot_path=joinpath(plot_path, "distribution"))
end

# Create posterior summary files (posterior_stats.csv, posterior_trajectory.csv) based on MCMC samples
nn_model, p_init, st = prepare_model(hyperpar, 1, sample_mechanistic, noise_model, true, component_vector=false)
create_posterior_summary_files(experiment_path, mcmc_res, 4096, p_init) # as many as we used as base for ensemble based UQ

parameters = mcmc_res.value[:,1:end-1,:]
if occursin("variational", experiment_series)
    save(joinpath(experiment_path, "parameters.jld2"), "parameters", cat(parameters.data[:,1:end,[1]], parameters.data[:,1:end-1,[2]], dims=1), "names", sample_names(pt)[1:end-1])
else
    save(joinpath(experiment_path, "parameters.jld2"), "parameters", parameters.data[:,1:end,:], "names", sample_names(pt)[1:end-1])
end
par_summary = constant_parameter_summary(mcmc_res)
save(joinpath(plot_path, "distribution", "summary.png"), par_summary)

#========================================================================================================
    Trace Plots and Density Plots
========================================================================================================#
trace_plot("alpha", parameters, 1, retrieve_α, 1:size(parameters)[1], [1])
trace_plot("gamma", parameters, 2, retrieve_γ, 1:size(parameters)[1], [1])
trace_plot("σ", parameters, 3, x -> sqrt(exp(x)), 1:size(parameters)[1], [1])
trace_plot("nn_1", parameters, 4, identity, 1:size(parameters)[1], [1])

pred = CSV.read(joinpath(experiment_path, "posterior_trajectory.csv"), DataFrame)
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
CSV.write(joinpath(experiment_path, "aggregate_predictions.csv"), pred_agg)
fig = plot_summary_overview(pred_agg)
save(joinpath(plot_path, "trajectory_summary.png"), fig)

