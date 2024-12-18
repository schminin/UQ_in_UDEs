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
# local settings
experiment_series = "constant_beta"
problem_name = "seir"
experiment_name = "multistart_gaussian_IR_noise_0.01"

# relative imports
experiment_path = joinpath(@__DIR__, problem_name, experiment_name)
include(joinpath(experiment_path, "experimental_setting.jl")) 
include("$problem_name.jl")
include("../../simulate.jl")
include("utils.jl")
include("evaluation_utils.jl")
include("../../evaluation_utils.jl")

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
rename!(pred, :S_opt => :S, :E_opt => :E, :I_opt => :I, :R_opt => :R, :β_opt => :β)

unique(pred.data_id)


function get_cutoff(df, p_chi)
    return quantile(Distributions.Chisq(1),1-p_chi)/2 + minimum(skipmissing(df.negLL_obs_p_opt))
end


pred_agg = aggregate_predictions(df_stats, pred, 1, problem_name, 0.05)
CSV.write(joinpath(experiment_path, "aggregated_predictions.csv"), pred_agg)