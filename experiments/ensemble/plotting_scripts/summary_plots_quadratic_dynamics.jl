#========================================================================================================
    1. summary figure for the states 
    2. summary figure for parameters
========================================================================================================#

using OrdinaryDiffEq, ModelingToolkit, SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using LinearAlgebra, Statistics
using DifferentialEquations # for automatic solver selection
using ComponentArrays, Lux, Zygote, Plots, StableRNGs, FileIO
using Distributions
using DataFrames, CSV
# gr()
using CairoMakie
using ColorSchemes

#========================================================================================================
    Experimental Settings
========================================================================================================#
# cluster settings
problem_name = ARGS[2] # "quadratic_dynamics"
experiment_name = ARGS[3] # "ground_truth_gaussian_noise_0.01"

# local settings
problem_name = "quadratic_dynamics"
experiment_series = "ensemble_10000"

# relative imports
include(joinpath(@__DIR__, "../$problem_name.jl"))
include("evaluation_utils.jl")


#========================================================================================================
    plots
========================================================================================================#
experiment_name = "multistart_gaussian_noise_0.05"

# set paths
experiment_path = joinpath(@__DIR__, experiment_series, problem_name, experiment_name)
ground_truth_path = joinpath(@__DIR__, "ground_truth", problem_name, replace(experiment_name, "multistart"=>"ground_truth"))
include(joinpath(experiment_path, "experimental_setting.jl")) 
include("../simulate.jl")
include("utils.jl")

# load ensemble results
df_trajectory = CSV.read(joinpath(experiment_path, "ensemble_trajectory.csv"), DataFrame)
df_stats = CSV.read(joinpath(experiment_path, "ensemble_stats.csv"), DataFrame)

df_trajectory_ground_truth = CSV.read(joinpath(ground_truth_path, "ensemble_trajectory.csv"), DataFrame)
df_stats_ground_truth = CSV.read(joinpath(ground_truth_path, "ensemble_stats.csv"), DataFrame)

# Define cutoff
p_chi = 0.95
cutoff_name = "chi_2_95"
cutoff_chi_95 = quantile(Distributions.Chisq(1),p_chi)/2 + minimum(skipmissing(df_stats.negLL_obs_p_opt))
df_sub = dropmissing(df_stats, :negLL_obs_p_opt)
df_sub = df_sub[df_sub.negLL_obs_p_opt .<= cutoff_chi_95, :]

cutoff_chi_95_ground_truth = quantile(Distributions.Chisq(1),0.95)/2 + minimum(skipmissing(df_stats_ground_truth.negLL_obs_p_opt))
df_sub_ground_truth = dropmissing(df_stats_ground_truth, :negLL_obs_p_opt)
df_sub_ground_truth = df_sub_ground_truth[df_sub_ground_truth.negLL_obs_p_opt .<= cutoff_chi_95_ground_truth, :]

# Create plot
# step 1: get model predictions
cutoff_models = df_sub.model_id
df_traj_sub = df_trajectory[[x in cutoff_models for x in df_trajectory.model_id],:]
gdf = groupby(df_traj_sub, :t)
traj_comb = combine(gdf, 
    :x_opt => minimum, :x_opt => mean, :x_opt => maximum)

df_sub = sort(df_sub, :negLL_obs_p_opt)
p = 75
cutoff_models_p = df_sub[1:Int(round(nrow(df_sub)*p/100)), :].model_id
df_traj_sub_p = df_trajectory[[x in cutoff_models_p for x in df_trajectory.model_id],:]
gdf_p = groupby(df_traj_sub_p, :t)
traj_comb_p = combine(gdf_p, 
    :x_opt => minimum, :x_opt => mean, :x_opt => maximum)

# step 2: get reference lines
ref = solve(ODEProblem(simulation_dynamics!, u0, tspan, p_ode), saveat = saveat_plot)

# step 3: get ground truth uncertainty
cutoff_models_ground_truth = df_sub_ground_truth.model_id
df_traj_sub_ground_truth = df_trajectory_ground_truth[[x in cutoff_models_ground_truth for x in df_trajectory_ground_truth.model_id],:]
gdf_ground_truth = groupby(df_traj_sub_ground_truth, :t)
traj_comb_ground_truth = combine(gdf_ground_truth, 
    :x_opt => minimum, :x_opt => mean, :x_opt => maximum)

summary_plot_path = joinpath(experiment_path, "plots", cutoff_name, "summary")
isdir(summary_plot_path) || mkpath(summary_plot_path)

##########################################################################
custom_colors = palette(:tab10)[1:5] # get(ColorSchemes.viridis, [i/(n_colors-1) for i in 0:1:(n_colors-1)])
trajectory_summary = let 
    f = Figure(size = (900, 400))

    ga = f[1, 1] = GridLayout()

    # Define axis of first line
    ax11 = CairoMakie.Axis(ga[1, 1], title = "A", backgroundcolor=(:white), xlabelsize=20, ylabel="x", width=200, height=200) #, width=200)
    ax12 = CairoMakie.Axis(ga[1, 2], title = "B", backgroundcolor=(:white), yticklabelsvisible=true, width=200, height=200) #, width=200)      
    ax13 = CairoMakie.Axis(ga[1, 3], title = "C", backgroundcolor=(:white), yticklabelsvisible=true, width=200, height=200) #, width=200)      
    ax14 = CairoMakie.Axis(ga[1, 4], title = "D", backgroundcolor=(:white), yticklabelsvisible=true, width=200, height=200) #, width=200)

    #linkyaxes!(ax11, ax12, ax13, ax14)
    linkxaxes!(ax11, ax12, ax13, ax14)

    # plot 1
    b11 = band!(ax11, traj_comb.t, traj_comb.x_opt_minimum, traj_comb.x_opt_maximum, alpha=0.0, color=(custom_colors[5],0.5))
    l11 = lines!(ax11, traj_comb.t, traj_comb.x_opt_mean, color=(custom_colors[5],1.0), linewidth=3)
    r11 = lines!(ax11, ref.t, Array(ref)[1,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)
    
    # plot 2
    b12 = band!(ax12, traj_comb.t, traj_comb.x_opt_minimum, traj_comb.x_opt_maximum, alpha=0.0, color=(custom_colors[5],0.5))
    l12 = lines!(ax12, traj_comb.t, traj_comb.x_opt_mean, color=(custom_colors[5],1.0), linewidth=3)
    gtb12 = band!(ax12, traj_comb_ground_truth.t, traj_comb_ground_truth.x_opt_minimum, traj_comb_ground_truth.x_opt_maximum, alpha=0.0, color=(:grey,0.5))
    gtl12 = lines!(ax12, traj_comb_ground_truth.t, traj_comb_ground_truth.x_opt_mean, color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

    legend15 = Legend(ga[2,1],
        [[r11]], # [hist, vline],
        ["reference"], orientation = :vertical, framevisible=false)

    legend25 = Legend(ga[2, 2],
        [[gtl12], [gtb12]], # [hist, vline],
        ["ground truth mean", "ground truth area"], orientation = :vertical, framevisible=false)

    
    ####### line 3
    b31 = band!(ax13, traj_comb.t, traj_comb.x_opt_minimum, traj_comb.x_opt_maximum, alpha=0.0, color=(custom_colors[5],0.5))
    l31 = lines!(ax13, traj_comb.t, traj_comb.x_opt_mean, color=(custom_colors[5],1.0), linewidth=3)
    gtb31 = band!(ax13, traj_comb_p.t, traj_comb_p.x_opt_minimum, traj_comb_p.x_opt_maximum, alpha=0.0, color=(custom_colors[5]-RGB(0.3,0.3,0.3),0.5))
    gtl31 = lines!(ax13, traj_comb_p.t, traj_comb_p.x_opt_mean, color=custom_colors[5]-RGB(0.3,0.3,0.3), linewidth=2, linestyle=:dash)

    legend35 = Legend(ga[2, 3],
        [[gtl31], [gtb31]], # [hist, vline],
        ["$p% ensemble mean", "$p% ensemble area"], orientation = :vertical, framevisible=false)
    f

    ############# line 4
    best_models = sort(df_sub, :negLL_obs_p_opt).model_id[1:10]

    l41 = lines!(ax14, traj_comb.t, traj_comb.x_opt_mean, color=(custom_colors[5],1.0), linewidth=3)
    r41 = lines!(ax14, ref.t, Array(ref)[1,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)
    for model_id in best_models
        df_trajectory_id = df_trajectory[df_trajectory.model_id.==Int(model_id), :]
        global l41s1 = lines!(ax14, df_trajectory_id.t, df_trajectory_id.x_opt, color=custom_colors[5]-RGB(0.3,0.3,0.3), linewidth=1)
    end


    legend45 = Legend(ga[2, 4],
        [[l41s1], [r41]], # [hist, vline],
        ["ensemble models", "reference"], orientation = :vertical, framevisible=false)

    legend51 = Legend(ga[1,5], [[l11], [b11]],
        ["ensemble mean", "ensemble area"], framevisible=false)

    resize_to_layout!(f)

    f
end

save(joinpath(summary_plot_path, "states_overview.png"), trajectory_summary)


##########################################################################
df_sub = dropmissing(df_stats, :negLL_obs_p_opt)
df_sub = df_sub[df_sub.negLL_obs_p_opt .<= cutoff_chi_95, :]

constant_parameters_summary = let
    f = Figure(size = (1000, 300))
    ga = f[1, 1] = GridLayout()

    # Distribution of alpha
    ax1 = CairoMakie.Axis(ga[1, 1], title = "α", backgroundcolor=(:white), xlabelsize=20)
    dens1 = CairoMakie.density!(ax1, df_sub.α_opt, normalization = :pdf, 
            label_size = 15, strokecolor = (:white, 0.5), color=(:steelblue4, 0.7)) 
    vline = vlines!(ax1, [p_ode[1]]; color=:tomato4, linewidth=2)
    CairoMakie.ylims!(low=0)

    if noise_model=="Gaussian"
        global par_name = "σ"
    elseif noise_model=="NegBin"
        global par_name = "overdispersion parameter"
    end
    ax3 = CairoMakie.Axis(ga[1, 2], title = par_name, backgroundcolor=(:white), xlabelsize=20)
    dens3 = CairoMakie.density!(ax3, df_sub.np_opt, normalization = :pdf, 
            label_size = 15, strokecolor = (:white, 0.5), color=(:lightblue, 0.7)) 
    vline = vlines!(ax3, [noise_par]; color=:tomato4, linewidth=2)
    CairoMakie.ylims!(low=0)

    Legend(ga[1, 3],
        [[dens1, dens3], vline],
        ["distribution", "reference"])
    f
end
save(joinpath(summary_plot_path, "const_parameters_overview.png"), constant_parameters_summary)

