using OrdinaryDiffEq, ModelingToolkit, SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using LinearAlgebra, Statistics
using DifferentialEquations # for automatic solver selection
using ComponentArrays, Lux, Zygote, Plots, StableRNGs, FileIO
using Distributions
using DataFrames, CSV
# gr()
# using CairoMakie
# using ColorSchemes

#========================================================================================================
    Experimental Settings
========================================================================================================#
# cluster settings
#problem_name = ARGS[2] # "quadratic_dynamics"
#experiment_name = ARGS[3] # "ground_truth_gaussian_noise_0.01"
#experiment_series = ARGS[4]

# local settings
experiment_series = "ensemble_10000/"
problem_name = "seir"
experiment_name = "multistart_gaussian_IR_noise_0.01"

# relative imports
ensemble_path = joinpath(@__DIR__,"../ensemble", experiment_series, problem_name, experiment_name)
include(joinpath(ensemble_path, "experimental_setting.jl")) 
# overwrite IC
include(joinpath(problem_name, replace(experiment_name, "multistart_" => ""), "experimental_setting.jl"))

include(joinpath(@__DIR__, "../$problem_name.jl"))
include("../simulate.jl")
include("../utils.jl")
include("../evaluation_utils.jl")
include("../ensemble/evaluation_utils.jl")

result_path = joinpath(@__DIR__, problem_name, replace(experiment_name, "multistart_" => ""), "ensemble", experiment_series)


#========================================================================================================
    Model Preprocessing
========================================================================================================#

"""
sol = solve(simulation_prob, Tsit5(), abstol = 1e-12, reltol = 1e-12, saveat = 0.001)
t = sol.t
X = Array(sol)

using Plots
plot(t, X')
"""

stats = CSV.read(joinpath(ensemble_path, "ensemble_stats.csv"), DataFrame)
function get_cutoff(df, p_chi)
    return quantile(Distributions.Chisq(1),1-p_chi)/2 + minimum(skipmissing(df.negLL_obs_p_opt))
end
cutoff_chi_2 = get_cutoff(stats, 0.05)
ensemble_members = stats[stats.negLL_obs_p_opt .<= cutoff_chi_2,"model_id"]

function calculate_model_trajectory(ensemble_path, model_id)
    nn_model, p_init, st = prepare_model(hyperpar, model_id, sample_mechanistic, noise_model, random_init)
    IC = u0
    prob_ude = ODEProblem(dynamics!, IC, tspan, p_init)
    function predict(p, saveat = t_train)  
        _prob = remake(prob_ude, p = p)
        Array(solve(_prob, Tsit5(), abstol = 1e-8, reltol = 1e-8, saveat = saveat))
    end
    p_opt = load(joinpath(ensemble_path, "models", "model_$(model_id)_p_opt.jld2"), "p_opt")
    pred_plot = predict(p_opt, tspan[1]:saveat_plot:tspan[end])
    df_trajectory = DataFrame(t = tspan[1]:saveat_plot:tspan[end], S_opt = pred_plot[1,:], E_opt = pred_plot[2,:], 
        I_opt = pred_plot[3,:], R_opt = pred_plot[4,:], β_opt = retrieve_β.(nn_model((tspan[1]:saveat_plot:tspan[end])', p_opt.nn, st)[1][1,:]),
         model_id = repeat([model_id], length(pred_plot[1,:])))
    return df_trajectory
end

# store model statistics of each ensemble model
df_trajectory = DataFrame()
data_id = 1
for model_id in ensemble_members
    try
        # println(model_id)
        local df_trajectory_sub = calculate_model_trajectory(ensemble_path, Int(model_id))
        if nrow(df_trajectory_sub)>0
            append!(df_trajectory, df_trajectory_sub, cols = :union)
        end
    catch
    end
end

if occursin("seir", problem_name)
    rename!(df_trajectory, :S_opt => :S, :E_opt => :E, :I_opt => :I, :R_opt => :R, :β_opt => :β)
elseif occursin("quadratic_dynamics", problem_name)
    rename!(df_trajectory, :x_opt => :x)
end

try
    unique(df_trajectory.data_id)
catch
    df_trajectory[!,"data_id"] = repeat([1], size(df_trajectory, 1))
end

# store results
CSV.write(joinpath(result_path, "ensemble_trajectory.csv"), df_trajectory)

df_trajectory = CSV.read(joinpath(result_path, "ensemble_trajectory.csv"), DataFrame)
pred_aggregated = seir_aggregation(df_trajectory)
CSV.write(joinpath(result_path, "aggregated_predictions.csv"), pred_aggregated)
