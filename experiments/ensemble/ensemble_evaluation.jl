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
problem_name = ARGS[2] # "quadratic_dynamics"
experiment_name = ARGS[3] # "ground_truth_gaussian_noise_0.01"
experiment_series = ARGS[4]

# local settings
experiment_series = "ensemble_10000/"
problem_name = "seir_pulse"
experiment_name = "multistart_negbin_IR_1.2"

# relative imports
experiment_path = joinpath(@__DIR__, experiment_series, problem_name, experiment_name)
include(joinpath(experiment_path, "experimental_setting.jl")) 
include(joinpath(@__DIR__, "../$problem_name.jl"))
include("../simulate.jl")
include("../utils.jl")
include("../evaluation_utils.jl")
include("evaluation_utils.jl")

#========================================================================================================
    Model Preprocessing
========================================================================================================#

# store model statistics of each ensemble model
df_stats = DataFrame()
df_trajectory = DataFrame()
for data_id in 1:aleatoric_noise_realizations
    for model_id in 1:ensemble_members
        try
            # println(model_id)
            local (col_names, col_entry, pred_data) = load_and_evaluate_model(experiment_path, data_id, model_id, hyperpar)
            append!(df_stats, DataFrame(col_entry', Symbol.(col_names)), cols=:union)
            if length(pred_data)>0
                append!(df_trajectory, tidy_prediction_data(data_id, model_id, pred_data), cols = :union)
            end
        catch
        end
    end
end
# store results
CSV.write(joinpath(experiment_path, "ensemble_stats.csv"), df_stats)
CSV.write(joinpath(experiment_path, "ensemble_trajectory.csv"), df_trajectory)

pred = CSV.read(joinpath(experiment_path, "ensemble_trajectory.csv"), DataFrame)
stats = CSV.read(joinpath(experiment_path, "ensemble_stats.csv"), DataFrame)

if occursin("seir", problem_name)
    rename!(pred, :S_opt => :S, :E_opt => :E, :I_opt => :I, :R_opt => :R, :β_opt => :β)
elseif occursin("quadratic_dynamics", problem_name)
    rename!(pred, :x_opt => :x)
end

try
    unique(pred.data_id)
catch
    pred[!,"data_id"] = repeat([1], size(pred, 1))
    stats[!,"data_id"] = repeat([1], size(stats, 1))
end

function get_cutoff(df, p_chi)
    return quantile(Distributions.Chisq(1),1-p_chi)/2 + minimum(skipmissing(df.negLL_obs_p_opt))
end

if length(unique(pred.data_id))>1
    # This is the ground truth ensemble with multiple data_ids
    pred_agg_total = DataFrame()
    for data_id in unique(pred.data_id)
        pred_agg = aggregate_predictions(stats, pred, data_id, problem_name, 0.05)
        pred_agg.data_id = repeat([data_id], size(pred_agg, 1))
        pred_agg_total = vcat(pred_agg_total, pred_agg)
    end
    CSV.write(joinpath(experiment_path, "aggregated_predictions_per_data_id.csv"), pred_agg_total)
    pred_agg_comb = combine(groupby(pred_agg_total, "t"),
        :S_minimum => minimum => :S_minimum, :S_maximum => maximum => :S_maximum, 
        :E_minimum => minimum => :E_minimum, :E_maximum => maximum => :E_maximum,
        :I_minimum => minimum => :I_minimum, :I_maximum => maximum => :I_maximum,
        :R_minimum => minimum => :R_minimum, :R_maximum => maximum => :R_maximum,
    )
    CSV.write(joinpath(experiment_path, "aggregated_predictions.csv"), pred_agg_comb)
else
    # This is the standard experimental setting
    pred_agg = aggregate_predictions(stats, pred, 1, problem_name, 0.05)
    CSV.write(joinpath(experiment_path, "aggregated_predictions.csv"), pred_agg)
end

#stats_sub = stats[stats.data_id.==1,:]
#pred_sub = pred[pred.data_id.==1,:]
#pred_agg = aggregate_predictions(stats_sub, pred_sub, 1, problem_name, 0.05)