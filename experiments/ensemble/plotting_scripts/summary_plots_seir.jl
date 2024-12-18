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
problem_name = ARGS[2] # problem_name = "seir"
experiment_name = ARGS[3] # experiment_name = "multistart_gaussian_IR_noise_0.01"

# local settings
experiment_series = "ensemble_10000"

# relative imports
include(joinpath("../../$problem_name.jl"))
include("../../evaluation_utils.jl")


#========================================================================================================
    plots
========================================================================================================#
# set paths
experiment_path = joinpath(@__DIR__, "..", experiment_series, problem_name, experiment_name)
ground_truth_path = joinpath(@__DIR__, "..", "ensemble_ground_truth", problem_name, replace(experiment_name, "multistart"=>"ground_truth"))
include(joinpath(experiment_path, "experimental_setting.jl")) 
include("../../simulate.jl")
include("../../utils.jl")

# load ensemble results
df_trajectory = CSV.read(joinpath(experiment_path, "ensemble_trajectory.csv"), DataFrame)
df_stats = CSV.read(joinpath(experiment_path, "ensemble_stats.csv"), DataFrame)

df_trajectory_ground_truth = CSV.read(joinpath(ground_truth_path, "ensemble_trajectory.csv"), DataFrame)
df_stats_ground_truth = CSV.read(joinpath(ground_truth_path, "ensemble_stats.csv"), DataFrame)

# Define cutoff

function apply_cutoff(df, p_chi)
    cutoff_chi_95 = quantile(Distributions.Chisq(1),p_chi)/2 + minimum(skipmissing(df.negLL_obs_p_opt))
    df_sub = dropmissing(df, :negLL_obs_p_opt)
    df_sub = df_sub[df_sub.negLL_obs_p_opt .<= cutoff_chi_95, :]
end

cutoff_name = "chi_2_95"
p_chi = 0.95
df_sub = apply_cutoff(df_stats, p_chi)

df_sub_ground_truth = DataFrame()
for data_id in unique(df_stats_ground_truth.data_id)
    append!(df_sub_ground_truth, apply_cutoff(df_stats_ground_truth[df_stats_ground_truth.data_id .== data_id, :], p_chi))
end
# we only select those data_ids with a reasonable enough minimal negLL
df_subselection_ground_truth = combine(groupby(df_sub_ground_truth, :data_id), :negLL_obs_p_opt => minimum, :negLL_obs_p_opt => maximum, :negLL_obs_p_opt => mean)
df_subselection_ground_truth = df_subselection_ground_truth[df_subselection_ground_truth.negLL_obs_p_opt_minimum .<= -2.9, :data_id]
df_sub_ground_truth = df_sub_ground_truth[[x in df_subselection_ground_truth for x in df_sub_ground_truth.data_id],:]

# Create plot
# step 1: get model predictions
cutoff_models = df_sub.model_id
df_traj_sub = df_trajectory[[x in cutoff_models for x in df_trajectory.model_id],:]
gdf = groupby(df_traj_sub, :t)
traj_comb = combine(gdf, 
    :S_opt => minimum, :S_opt => mean, :S_opt => maximum,
    :E_opt => minimum, :E_opt => mean, :E_opt => maximum,
    :I_opt => minimum, :I_opt => mean, :I_opt => maximum,
    :R_opt => minimum, :R_opt => mean, :R_opt => maximum,
    :β_opt => minimum, :β_opt => mean, :β_opt => maximum)

df_sub = sort(df_sub, :negLL_obs_p_opt)
p = 75
cutoff_models_p = df_sub[1:Int(round(nrow(df_sub)*p/100)), :].model_id
df_traj_sub_p = df_trajectory[[x in cutoff_models_p for x in df_trajectory.model_id],:]
gdf_p = groupby(df_traj_sub_p, :t)
traj_comb_p = combine(gdf_p, 
    :S_opt => minimum, :S_opt => mean, :S_opt => maximum,
    :E_opt => minimum, :E_opt => mean, :E_opt => maximum,
    :I_opt => minimum, :I_opt => mean, :I_opt => maximum,
    :R_opt => minimum, :R_opt => mean, :R_opt => maximum,
    :β_opt => minimum, :β_opt => mean, :β_opt => maximum)

# step 2: get reference lines
ref = solve(ODEProblem(simulation_dynamics!, u0, tspan, p_ode), saveat = saveat_plot)

# step 3: get ground truth uncertainty
function accepted_trajectory(model_id, data_id; df_sub_ground_truth = df_sub_ground_truth)
    cond = (df_sub_ground_truth.model_id .== model_id) .&& (df_sub_ground_truth.data_id .== data_id)
    if sum(cond) == 1
        return true
    else
        return false
    end
end

df_trajectory_ground_truth = df_trajectory_ground_truth[[x in df_subselection_ground_truth for x in df_trajectory_ground_truth.data_id],:]
cond = accepted_trajectory.(df_trajectory_ground_truth.model_id,df_trajectory_ground_truth.data_id)

df_traj_sub_ground_truth = df_trajectory_ground_truth[cond,:]
gdf_ground_truth = groupby(df_traj_sub_ground_truth, :t)
traj_comb_ground_truth = combine(gdf_ground_truth, 
    :S_opt => minimum, :S_opt => mean, :S_opt => maximum,
    :E_opt => minimum, :E_opt => mean, :E_opt => maximum,
    :I_opt => minimum, :I_opt => mean, :I_opt => maximum,
    :R_opt => minimum, :R_opt => mean, :R_opt => maximum,
    :β_opt => minimum, :β_opt => mean, :β_opt => maximum)


summary_plot_path = joinpath(experiment_path, "plots", cutoff_name, "summary")
isdir(summary_plot_path) || mkpath(summary_plot_path)

##########################################################################
custom_colors = palette(:tab10)[1:4] # get(ColorSchemes.viridis, [i/(n_colors-1) for i in 0:1:(n_colors-1)])
trajectory_summary = let 
    f = Figure(size = (1500, 700))

    ga = f[1, 1] = GridLayout()

    # Define axis of first line
    ax11 = CairoMakie.Axis(ga[1, 2], title = "S", backgroundcolor=(:white), xlabelsize=20, 
        xticklabelsvisible=false) #, width=200)
    ax12 = CairoMakie.Axis(ga[1, 3], title = "E", backgroundcolor=(:white), xticklabelsvisible=false, yticklabelsvisible=false) #, width=200)      
    ax13 = CairoMakie.Axis(ga[1, 4], title = "I", backgroundcolor=(:white), xticklabelsvisible=false, yticklabelsvisible=false) #, width=200)      
    ax14 = CairoMakie.Axis(ga[1, 5], title = "R", backgroundcolor=(:white), xticklabelsvisible=false, yticklabelsvisible=false) #, width=200)

    ax1 = CairoMakie.Axis(ga[1,1], width = 60)
    hidedecorations!(ax1)  # hides ticks, grid and lables
    hidespines!(ax1)
    text!(ax1, "A", fontsize=25,space = :relative, font=:bold, position=Point2f(0.,0.7))
    ax2 = CairoMakie.Axis(ga[2,1], width = 60)
    hidedecorations!(ax2)  # hides ticks, grid and lables
    hidespines!(ax2)
    text!(ax2, "B", fontsize=25,space = :relative, font=:bold, position=Point2f(0.,0.7))
    ax3 = CairoMakie.Axis(ga[3,1], width = 60)
    hidedecorations!(ax3)  # hides ticks, grid and lables
    hidespines!(ax3)
    text!(ax3, "C", fontsize=25,space = :relative, font=:bold, position=Point2f(0.,0.7))
    ax4 = CairoMakie.Axis(ga[4,1], width = 60)
    hidedecorations!(ax4)  # hides ticks, grid and lables
    hidespines!(ax4)
    text!(ax4, "D", fontsize=25,space = :relative, font=:bold, position=Point2f(0.,0.7))
    f
    # Define axis of second line
    ax21 = CairoMakie.Axis(ga[2, 2], backgroundcolor=(:white), xticklabelsvisible=false) #, width=200)
    ax22 = CairoMakie.Axis(ga[2, 3], backgroundcolor=(:white), xticklabelsvisible=false, yticklabelsvisible=false) #, width=200)      
    ax23 = CairoMakie.Axis(ga[2, 4], backgroundcolor=(:white), xticklabelsvisible=false, yticklabelsvisible=false) #, width=200)      
    ax24 = CairoMakie.Axis(ga[2, 5], backgroundcolor=(:white), xticklabelsvisible=false, yticklabelsvisible=false) #, width=200) 

    
    # Define axis of third line
    ax31 = CairoMakie.Axis(ga[3, 2], backgroundcolor=(:white), xticklabelsvisible=false)
    ax32 = CairoMakie.Axis(ga[3, 3], backgroundcolor=(:white), xticklabelsvisible=false, yticklabelsvisible=false)      
    ax33 = CairoMakie.Axis(ga[3, 4], backgroundcolor=(:white), xticklabelsvisible=false, yticklabelsvisible=false)      
    ax34 = CairoMakie.Axis(ga[3, 5], backgroundcolor=(:white), xticklabelsvisible=false, yticklabelsvisible=false)  

    # Define axis of forth line
    ax41 = CairoMakie.Axis(ga[4, 2], xlabel = "time", backgroundcolor=(:white))
    ax42 = CairoMakie.Axis(ga[4, 3], xlabel = "time", backgroundcolor=(:white), yticklabelsvisible=false)      
    ax43 = CairoMakie.Axis(ga[4, 4], xlabel = "time", backgroundcolor=(:white), yticklabelsvisible=false)      
    ax44 = CairoMakie.Axis(ga[4, 5], xlabel = "time", backgroundcolor=(:white), yticklabelsvisible=false)      

    linkxaxes!(ax11, ax21, ax31, ax41)
    linkxaxes!(ax12, ax22, ax32, ax42)
    linkxaxes!(ax13, ax23, ax33, ax43)
    linkxaxes!(ax14, ax24, ax34, ax44)

    linkyaxes!(ax11, ax12, ax13, ax14)
    linkyaxes!(ax21, ax22, ax23, ax24)
    linkyaxes!(ax31, ax32, ax33, ax34)
    linkyaxes!(ax41, ax42, ax43, ax44)

    linkyaxes!(ax41, ax42, ax43, ax44)


    # Define content of first line
    b11 = band!(ax11, traj_comb.t, traj_comb.S_opt_minimum, traj_comb.S_opt_maximum, alpha=0.0, color=(custom_colors[1],0.5))
    l11 = lines!(ax11, traj_comb.t, traj_comb.S_opt_mean, color=(custom_colors[1],1.0), linewidth=3)
    r11 = lines!(ax11, ref.t, Array(ref)[1,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

    b12 = band!(ax12, traj_comb.t, traj_comb.E_opt_minimum, traj_comb.E_opt_maximum, alpha=0.01, color=(custom_colors[2],0.5)) 
    l12 = lines!(ax12, traj_comb.t, traj_comb.E_opt_mean, color=(custom_colors[2],1.0), linewidth=3)
    r12 = lines!(ax12, ref.t, Array(ref)[2,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

    b13 = band!(ax13, traj_comb.t, traj_comb.I_opt_minimum, traj_comb.I_opt_maximum, alpha=0.01, color=(custom_colors[3],0.5)) 
    l13 = lines!(ax13, traj_comb.t, traj_comb.I_opt_mean, color=(custom_colors[3],1.0), linewidth=3)
    s13 = CairoMakie.scatter!(ax13, t_sim, X_sim[3,:], color = (custom_colors[3]-RGB(0.3,0.3,0.3),1), markersize=5)
    r13 = lines!(ax13, ref.t, Array(ref)[3,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

    b14 = band!(ax14, traj_comb.t, traj_comb.R_opt_minimum, traj_comb.R_opt_maximum, alpha=0.01, color=(custom_colors[4],0.5)) 
    l14 = lines!(ax14, traj_comb.t, traj_comb.R_opt_mean, color=(custom_colors[4],1.0), linewidth=3)
    s14 = CairoMakie.scatter!(ax14, t_sim, X_sim[4,:], color = (custom_colors[4]-RGB(0.3,0.3,0.3),1), markersize=5)
    r14 = lines!(ax14, ref.t, Array(ref)[4,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

    legend15 = Legend(ga[1, 6],
        [[r11, r12, r13, r14]], # [hist, vline],
        ["reference"], orientation = :vertical, framevisible=false)

    # Define content of second line
    b21 = band!(ax21, traj_comb.t, traj_comb.S_opt_minimum, traj_comb.S_opt_maximum, alpha=0.0, color=(custom_colors[1],0.5))
    l21 = lines!(ax21, traj_comb.t, traj_comb.S_opt_mean, color=(custom_colors[1],1.0), linewidth=3)
    gtb21 = band!(ax21, traj_comb_ground_truth.t, traj_comb_ground_truth.S_opt_minimum, traj_comb_ground_truth.S_opt_maximum, alpha=0.0, color=(:grey,0.5))
    gtl21 = lines!(ax21, traj_comb_ground_truth.t, traj_comb_ground_truth.S_opt_mean, color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

    b22 = band!(ax22, traj_comb.t, traj_comb.E_opt_minimum, traj_comb.E_opt_maximum, alpha=0.01, color=(custom_colors[2],0.5)) 
    l22 = lines!(ax22, traj_comb.t, traj_comb.E_opt_mean, color=(custom_colors[2],1.0), linewidth=3)
    gtb22 = band!(ax22, traj_comb_ground_truth.t, traj_comb_ground_truth.E_opt_minimum, traj_comb_ground_truth.E_opt_maximum, alpha=0.0, color=(:grey,0.5))
    gtl22 = lines!(ax22, traj_comb_ground_truth.t, traj_comb_ground_truth.E_opt_mean, color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

    b23 = band!(ax23, traj_comb.t, traj_comb.I_opt_minimum, traj_comb.I_opt_maximum, alpha=0.01, color=(custom_colors[3],0.5)) 
    l23 = lines!(ax23, traj_comb.t, traj_comb.I_opt_mean, color=(custom_colors[3],1.0), linewidth=3)
    s23 = CairoMakie.scatter!(ax23, t_sim, X_sim[3,:], color = (custom_colors[3]-RGB(0.3,0.3,0.3),1), markersize=5)
    gtb23 = band!(ax23, traj_comb_ground_truth.t, traj_comb_ground_truth.I_opt_minimum, traj_comb_ground_truth.I_opt_maximum, alpha=0.0, color=(:grey,0.5))
    gtl23 = lines!(ax23, traj_comb_ground_truth.t, traj_comb_ground_truth.I_opt_mean, color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

    b24 = band!(ax24, traj_comb.t, traj_comb.R_opt_minimum, traj_comb.R_opt_maximum, alpha=0.01, color=(custom_colors[4],0.5)) 
    s24 = CairoMakie.scatter!(ax24, t_sim, X_sim[4,:], color = (custom_colors[4]-RGB(0.3,0.3,0.3),1), markersize=5)
    l24 = lines!(ax24, traj_comb.t, traj_comb.R_opt_mean, color=(custom_colors[4],1.0), linewidth=3)
    gtb24 = band!(ax24, traj_comb_ground_truth.t, traj_comb_ground_truth.R_opt_minimum, traj_comb_ground_truth.R_opt_maximum, alpha=0.0, color=(:grey,0.5))
    gtl24 = lines!(ax24, traj_comb_ground_truth.t, traj_comb_ground_truth.R_opt_mean, color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

    legend25 = Legend(ga[2, 6],
        [[gtl21, gtl22, gtl23, gtl24], [gtb21, gtb22, gtb23, gtb24]], # [hist, vline],
        ["ground truth mean", "ground truth area"], orientation = :vertical, framevisible=false)

    ####### line 3
    b31 = band!(ax31, traj_comb.t, traj_comb.S_opt_minimum, traj_comb.S_opt_maximum, alpha=0.0, color=(custom_colors[1],0.5))
    l31 = lines!(ax31, traj_comb.t, traj_comb.S_opt_mean, color=(custom_colors[1],1.0), linewidth=3)
    gtb31 = band!(ax31, traj_comb_p.t, traj_comb_p.S_opt_minimum, traj_comb_p.S_opt_maximum, alpha=0.0, color=(custom_colors[1]-RGB(0.3,0.3,0.3),0.5))
    gtl31 = lines!(ax31, traj_comb_p.t, traj_comb_p.S_opt_mean, color=custom_colors[1]-RGB(0.3,0.3,0.3), linewidth=2, linestyle=:dash)

    b32 = band!(ax32, traj_comb.t, traj_comb.E_opt_minimum, traj_comb.E_opt_maximum, alpha=0.01, color=(custom_colors[2],0.5)) 
    l32 = lines!(ax32, traj_comb.t, traj_comb.E_opt_mean, color=(custom_colors[2],1.0), linewidth=3)
    gtb32 = band!(ax32, traj_comb_p.t, traj_comb_p.E_opt_minimum, traj_comb_p.E_opt_maximum, alpha=0.0, color=(custom_colors[2]-RGB(0.3,0.3,0.3),0.5))
    gtl32 = lines!(ax32, traj_comb_p.t, traj_comb_p.E_opt_mean, color=custom_colors[2]-RGB(0.3,0.3,0.3), linewidth=2, linestyle=:dash)

    b33 = band!(ax33, traj_comb.t, traj_comb.I_opt_minimum, traj_comb.I_opt_maximum, alpha=0.01, color=(custom_colors[3],0.5)) 
    l33 = lines!(ax33, traj_comb.t, traj_comb.I_opt_mean, color=(custom_colors[3],1.0), linewidth=3)
    s33 = CairoMakie.scatter!(ax33, t_sim, X_sim[3,:], color = (custom_colors[3]-RGB(0.3,0.3,0.3),1), markersize=5)
    gtb33 = band!(ax33, traj_comb_p.t, traj_comb_p.I_opt_minimum, traj_comb_p.I_opt_maximum, alpha=0.0, color=(custom_colors[3]-RGB(0.3,0.3,0.3),0.5))
    gtl33 = lines!(ax33, traj_comb_p.t, traj_comb_p.I_opt_mean, color=custom_colors[3]-RGB(0.3,0.3,0.3), linewidth=2, linestyle=:dash)

    b34 = band!(ax34, traj_comb.t, traj_comb.R_opt_minimum, traj_comb.R_opt_maximum, alpha=0.01, color=(custom_colors[4],0.5)) 
    s34 = CairoMakie.scatter!(ax34, t_sim, X_sim[4,:], color = (custom_colors[4]-RGB(0.3,0.3,0.3),1), markersize=5)
    l34 = lines!(ax34, traj_comb.t, traj_comb.R_opt_mean, color=(custom_colors[4],1.0), linewidth=3)
    gtb34 = band!(ax34, traj_comb_p.t, traj_comb_p.R_opt_minimum, traj_comb_p.R_opt_maximum, alpha=0.0, color=(custom_colors[4]-RGB(0.3,0.3,0.3),0.5))
    gtl34 = lines!(ax34, traj_comb_p.t, traj_comb_p.R_opt_mean, color=custom_colors[4]-RGB(0.3,0.3,0.3), linewidth=2, linestyle=:dash)

    legend35 = Legend(ga[3, 6],
        [[gtl31, gtl32, gtl33, gtl34], [gtb31, gtb32, gtb33, gtb34]], # [hist, vline],
        ["$p% ensemble mean", "$p% ensemble area"], orientation = :vertical, framevisible=false)


    ############# line 4
    best_models = sort(df_sub, :negLL_obs_p_opt).model_id[1:10]

    l41 = lines!(ax41, traj_comb.t, traj_comb.S_opt_mean, color=(custom_colors[1],1.0), linewidth=3)
    r41 = lines!(ax41, ref.t, Array(ref)[1,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)
    for model_id in best_models
        df_trajectory_id = df_trajectory[df_trajectory.model_id.==Int(model_id), :]
        global l41s1 = lines!(ax41, df_trajectory_id.t, df_trajectory_id.S_opt, color=custom_colors[1]-RGB(0.3,0.3,0.3), linewidth=1)
    end

    l42 = lines!(ax42, traj_comb.t, traj_comb.E_opt_mean, color=(custom_colors[2],1.0), linewidth=3)
    r42 = lines!(ax42, ref.t, Array(ref)[2,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)
    for model_id in best_models
        df_trajectory_id = df_trajectory[df_trajectory.model_id.==Int(model_id), :]
        global l42s2 = lines!(ax42, df_trajectory_id.t, df_trajectory_id.E_opt, color=custom_colors[2]-RGB(0.3,0.3,0.3), linewidth=1)
    end

    l43 = lines!(ax43, traj_comb.t, traj_comb.I_opt_mean, color=(custom_colors[3],1.0), linewidth=3)
    r43 = lines!(ax43, ref.t, Array(ref)[3,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)
    s43 = CairoMakie.scatter!(ax43, t_sim, X_sim[3,:], color = (custom_colors[3]-RGB(0.3,0.3,0.3),1), markersize=5)
    for model_id in best_models
        df_trajectory_id = df_trajectory[df_trajectory.model_id.==Int(model_id), :]
        global l43s3 = lines!(ax43, df_trajectory_id.t, df_trajectory_id.I_opt, color=custom_colors[3]-RGB(0.3,0.3,0.3), linewidth=1)
    end

    s44 = CairoMakie.scatter!(ax44, t_sim, X_sim[4,:], color = (custom_colors[4]-RGB(0.3,0.3,0.3),1), markersize=5)
    r44 = lines!(ax44, ref.t, Array(ref)[4,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)
    l44 = lines!(ax44, traj_comb.t, traj_comb.R_opt_mean, color=(custom_colors[4],1.0), linewidth=3)
    for model_id in best_models
        df_trajectory_id = df_trajectory[df_trajectory.model_id.==Int(model_id), :]
        global l44s4 = lines!(ax44, df_trajectory_id.t, df_trajectory_id.R_opt, color=custom_colors[4]-RGB(0.3,0.3,0.3), linewidth=1) 
    end

    legend45 = Legend(ga[4, 6],
        [[l41s1, l42s2, l43s3, l44s4], [r41, r42, r43, r44]], # [hist, vline],
        ["ensemble models", "reference"], orientation = :vertical, framevisible=false)

    legend51 = Legend(ga[5,2], [[l11, l21, l31, l41], [b11, b21, b31]],
        ["ensemble mean", "ensemble area"], framevisible=false)
    legend52 = Legend(ga[5,3], [[l12, l22, l32, l42], [b12, b22, b32]],
        ["ensemble mean", "ensemble area"], framevisible=false)
    legend53 = Legend(ga[5,4], [[l13, l23, l33, l43], [b13, b23, b33], [s13, s23, s33, s43]],
        ["ensemble mean", "ensemble area", "observed"], framevisible=false)
    legend54 = Legend(ga[5,5], [[l14, l24, l34, l44], [b14, b24, b34], [s14, s24, s34, s44]],
        ["ensemble mean", "ensemble area", "observed"], framevisible=false)

    resize_to_layout!(f)

    f
end

save(joinpath(summary_plot_path, "states_overview.png"), trajectory_summary)


##########################################################################

time_varying_parameter_summary = let 
    f = Figure(size = (900, 400))

    ga = f[1, 1] = GridLayout()

    # Define axis of first line
    ax11 = CairoMakie.Axis(ga[1, 1], title = "A", backgroundcolor=(:white), xlabelsize=20, ylabel="β", width=200, height=200) #, width=200)
    ax12 = CairoMakie.Axis(ga[1, 2], title = "B", backgroundcolor=(:white), yticklabelsvisible=false, width=200, height=200) #, width=200)      
    ax13 = CairoMakie.Axis(ga[1, 3], title = "C", backgroundcolor=(:white), yticklabelsvisible=false, width=200, height=200) #, width=200)      
    ax14 = CairoMakie.Axis(ga[1, 4], title = "D", backgroundcolor=(:white), yticklabelsvisible=false, width=200, height=200) #, width=200)

    linkyaxes!(ax11, ax12, ax13, ax14)
    linkxaxes!(ax11, ax12, ax13, ax14)

    # plot 1
    b11 = band!(ax11, traj_comb.t, traj_comb.β_opt_minimum, traj_comb.β_opt_maximum, alpha=0.0, color=(palette(:default)[5],0.5))
    l11 = lines!(ax11, traj_comb.t, traj_comb.β_opt_mean, color=(palette(:default)[5],1.0), linewidth=3)
    r11 = lines!(ax11, ref.t, β.(ref.t), color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)
    
    # plot 2
    b12 = band!(ax12, traj_comb.t, traj_comb.β_opt_minimum, traj_comb.β_opt_maximum, alpha=0.0, color=(palette(:default)[5],0.5))
    l12 = lines!(ax12, traj_comb.t, traj_comb.β_opt_mean, color=(palette(:default)[5],1.0), linewidth=3)
    gtb12 = band!(ax12, traj_comb_ground_truth.t, traj_comb_ground_truth.β_opt_minimum, traj_comb_ground_truth.β_opt_maximum, alpha=0.0, color=(:grey,0.5))
    gtl12 = lines!(ax12, traj_comb_ground_truth.t, traj_comb_ground_truth.β_opt_mean, color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

    legend15 = Legend(ga[2,1],
        [[r11]], # [hist, vline],
        ["reference"], orientation = :vertical, framevisible=false)

    legend25 = Legend(ga[2, 2],
        [[gtl12], [gtb12]], # [hist, vline],
        ["ground truth mean", "ground truth area"], orientation = :vertical, framevisible=false)

    
    ####### line 3
    b31 = band!(ax13, traj_comb.t, traj_comb.β_opt_minimum, traj_comb.β_opt_maximum, alpha=0.0, color=(palette(:default)[5],0.5))
    l31 = lines!(ax13, traj_comb.t, traj_comb.β_opt_mean, color=(palette(:default)[5],1.0), linewidth=3)
    gtb31 = band!(ax13, traj_comb_p.t, traj_comb_p.β_opt_minimum, traj_comb_p.β_opt_maximum, alpha=0.0, color=(palette(:default)[5]-RGB(0.3,0.3,0.3),0.5))
    gtl31 = lines!(ax13, traj_comb_p.t, traj_comb_p.β_opt_mean, color=palette(:default)[5]-RGB(0.3,0.3,0.3), linewidth=2, linestyle=:dash)

    legend35 = Legend(ga[2, 3],
        [[gtl31], [gtb31]], # [hist, vline],
        ["$p% ensemble mean", "$p% ensemble area"], orientation = :vertical, framevisible=false)
    f

    ############# line 4
    best_models = sort(df_sub, :negLL_obs_p_opt).model_id[1:10]

    l41 = lines!(ax14, traj_comb.t, traj_comb.β_opt_mean, color=(palette(:default)[5],1.0), linewidth=3)
    r41 = lines!(ax14, ref.t, β.(ref.t), color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)
    for model_id in best_models
        df_trajectory_id = df_trajectory[df_trajectory.model_id.==Int(model_id), :]
        global l41s1 = lines!(ax14, df_trajectory_id.t, df_trajectory_id.β_opt, color=palette(:default)[5]-RGB(0.3,0.3,0.3), linewidth=1)
    end


    legend45 = Legend(ga[2, 4],
        [[l41s1], [r41]], # [hist, vline],
        ["ensemble models", "reference"], orientation = :vertical, framevisible=false)

    legend51 = Legend(ga[1,5], [[l11], [b11]],
        ["ensemble mean", "ensemble area"], framevisible=false)

    resize_to_layout!(f)

    f
end

save(joinpath(summary_plot_path, "time_varying_parameter_overview.png"), time_varying_parameter_summary)

##########################################################################
df_sub = dropmissing(df_stats, :negLL_obs_p_opt)
cutoff_chi_95 = quantile(Distributions.Chisq(1),p_chi)/2 + minimum(skipmissing(df_sub.negLL_obs_p_opt))
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
    ax2 = CairoMakie.Axis(ga[1, 2], title = "γ", backgroundcolor=(:white), xlabelsize=20)
    dens2 = CairoMakie.density!(ax2, df_sub.γ_opt, normalization = :pdf, 
            label_size = 15, strokecolor = (:white, 0.5), color=(:darkslategray4, 0.7)) 
    vline = vlines!(ax2, [p_ode[2]]; color=:tomato4, linewidth=2)
    CairoMakie.ylims!(low=0)
    if noise_model=="Gaussian"
        global par_name = "σ"
    elseif noise_model=="negBin"
        global par_name = "overdispersion parameter"
    end
    ax3 = CairoMakie.Axis(ga[1, 3], title = par_name, backgroundcolor=(:white), xlabelsize=20)
    dens3 = CairoMakie.density!(ax3, df_sub.np_opt, normalization = :pdf, 
            label_size = 15, strokecolor = (:white, 0.5), color=(:lightblue, 0.7)) 
    vline = vlines!(ax3, [noise_par]; color=:tomato4, linewidth=2)
    CairoMakie.ylims!(low=0)

    Legend(ga[1, 4],
        [[dens1, dens2, dens3], vline],
        ["distribution", "reference"])
    f
end
save(joinpath(summary_plot_path, "const_parameters_overview.png"), constant_parameters_summary)

