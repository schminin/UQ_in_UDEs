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
problem_name = "seir"
experiment_name = "multistart_gaussian_IR_noise_0.01"
ground_truth = "ground_truth_gaussian_IR_noise_0.01"

experiment_path = joinpath(@__DIR__, "..", "experiments", "ensemble", "ensemble_10000", problem_name, experiment_name)
# ground_truth_path = joinpath(@__DIR__, "..", "experiments", "ensemble", "ensemble_ground_truth", problem_name, ground_truth)

df_trajectory = CSV.read(joinpath(experiment_path, "ensemble_trajectory.csv"), DataFrame)
df_stats = CSV.read(joinpath(experiment_path, "ensemble_stats.csv"), DataFrame)
pred_agg = CSV.read(joinpath(experiment_path, "aggregated_predictions.csv"), DataFrame)

# pred_agg_ground_truth = CSV.read(joinpath(ground_truth_path, "aggregated_predictions.csv"), DataFrame)

include(joinpath(experiment_path, "experimental_setting.jl")) 
include("../experiments/$(problem_name).jl")
include("../experiments/simulate.jl")
include("../experiments/utils.jl")
ref = solve(ODEProblem(simulation_dynamics!, u0, tspan, p_ode), saveat = saveat_plot)

custom_colors = palette(:tab10)[1:4] # get(ColorSchemes.viridis, [i/(n_colors-1) for i in 0:1:(n_colors-1)])
custom_bands = [RGB(165/255, 200/255, 225/255), RGB(255/255, 203/255, 158/255), RGB(170/255, 217/255, 170/255), RGB(238/255, 168/255, 169/255)]

best_models = sort(df_stats, :negLL_obs_p_opt).model_id[1:10]

if occursin("seir", problem_name)
    rename!(df_trajectory, :S_opt => :S, :E_opt => :E, :I_opt => :I, :R_opt => :R, :β_opt => :β)
elseif occursin("quadratic_dynamics", problem_name)
    rename!(df_trajectory, :x_opt => :x)
end

CairoMakie.activate!(type = "pdf")

#========================================================================================================
    Plot with
    - ensemble mean and area
    - ground truth mean and area
    - uncertainty shape
    - 10 best ensemble members
========================================================================================================#
trajectory_summary = let 
    f = Figure(size = (1500, 500),  px_per_unit = 10)

    ga = f[1, 1] = GridLayout()
    colgap!(ga, 1)
    rowgap!(ga, 1)

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
    
    ax3 = CairoMakie.Axis(ga[2,1], width = 60)
    hidedecorations!(ax3)  # hides ticks, grid and lables
    hidespines!(ax3)
    text!(ax3, "B", fontsize=25,space = :relative, font=:bold, position=Point2f(0.,0.7))
    ax4 = CairoMakie.Axis(ga[3,1], width = 60)
    hidedecorations!(ax4)  # hides ticks, grid and lables
    hidespines!(ax4)
    text!(ax4, "C", fontsize=25,space = :relative, font=:bold, position=Point2f(0.,0.7))
    f
    # Define axis of second line
    ax31 = CairoMakie.Axis(ga[2, 2], backgroundcolor=(:white), xticklabelsvisible=false) #, width=200)
    ax32 = CairoMakie.Axis(ga[2, 3], backgroundcolor=(:white), xticklabelsvisible=false, yticklabelsvisible=false) #, width=200)      
    ax33 = CairoMakie.Axis(ga[2, 4], backgroundcolor=(:white), xticklabelsvisible=false, yticklabelsvisible=false) #, width=200)      
    ax34 = CairoMakie.Axis(ga[2, 5], backgroundcolor=(:white), xticklabelsvisible=false, yticklabelsvisible=false) #, width=200) 

    # Define axis of forth line
    ax41 = CairoMakie.Axis(ga[3, 2], xlabel = "time", backgroundcolor=(:white))
    ax42 = CairoMakie.Axis(ga[3, 3], xlabel = "time", backgroundcolor=(:white), yticklabelsvisible=false)      
    ax43 = CairoMakie.Axis(ga[3, 4], xlabel = "time", backgroundcolor=(:white), yticklabelsvisible=false)      
    ax44 = CairoMakie.Axis(ga[3, 5], xlabel = "time", backgroundcolor=(:white), yticklabelsvisible=false)      

    linkxaxes!(ax11,  ax31, ax41)
    linkxaxes!(ax12, ax32, ax42)
    linkxaxes!(ax13,  ax33, ax43)
    linkxaxes!(ax14,  ax34, ax44)

    linkyaxes!(ax11, ax12, ax13, ax14)
    linkyaxes!(ax31, ax32, ax33, ax34)
    linkyaxes!(ax41, ax42, ax43, ax44)

    # Define content of first line
    b_legend = band!(ax11, pred_agg.t, pred_agg.S_q_p5, pred_agg.S_q_99p5, alpha=0.0, color=RGB(0.7,0.7,0.7))
    l_legend = lines!(ax11, pred_agg.t, pred_agg.S_mean, color=RGB(0.4, 0.4, 0.4), linewidth=3)
    s_legend = CairoMakie.scatter!(ax13, t_sim, X_sim[3,:], color=RGB(0.4, 0.4, 0.4), markersize=7)
    

    # Define content of first line
    b11 = band!(ax11, pred_agg.t, pred_agg.S_q_p5, pred_agg.S_q_99p5, alpha=0.0, color=custom_bands[1])
    l11 = lines!(ax11, pred_agg.t, pred_agg.S_mean, color=(custom_colors[1],1.0), linewidth=3)
    r11 = lines!(ax11, ref.t, Array(ref)[1,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

    b12 = band!(ax12, pred_agg.t, pred_agg.E_q_p5, pred_agg.E_q_99p5, alpha=0.01, color=(custom_colors[2],0.4)) 
    l12 = lines!(ax12, pred_agg.t, pred_agg.E_mean, color=(custom_colors[2],1.0), linewidth=3)
    r12 = lines!(ax12, ref.t, Array(ref)[2,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

    b13 = band!(ax13, pred_agg.t, pred_agg.I_q_p5, pred_agg.I_q_99p5, alpha=0.01, color=(custom_colors[3],0.4)) 
    l13 = lines!(ax13, pred_agg.t, pred_agg.I_mean, color=(custom_colors[3],1.0), linewidth=3)
    s13 = CairoMakie.scatter!(ax13, t_sim, X_sim[3,:], color = (custom_colors[3]-RGB(0.3,0.3,0.3),1), markersize=5)
    r13 = lines!(ax13, ref.t, Array(ref)[3,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

    b14 = band!(ax14, pred_agg.t, pred_agg.R_q_p5, pred_agg.R_q_99p5, alpha=0.01, color=(custom_colors[4],0.4)) 
    l14 = lines!(ax14, pred_agg.t, pred_agg.R_mean, color=(custom_colors[4],1.0), linewidth=3)
    s14 = CairoMakie.scatter!(ax14, t_sim, X_sim[4,:], color = (custom_colors[4]-RGB(0.3,0.3,0.3),1), markersize=5)
    r14 = lines!(ax14, ref.t, Array(ref)[4,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

    ####### line 2
    b31 = band!(ax31, pred_agg.t, pred_agg.S_q_p5, pred_agg.S_q_99p5, alpha=0.0, color=(custom_colors[1],0.4))
    l31 = lines!(ax31, pred_agg.t, pred_agg.S_mean, color=(custom_colors[1],1.0), linewidth=3)
    gtb31 = band!(ax31, pred_agg.t, pred_agg.S_q_10, pred_agg.S_q_90, alpha=0.0, color=(custom_colors[1]-RGB(0.1,0.1,0.1),0.5))
    gtb31_2 = band!(ax31, pred_agg.t, pred_agg.S_q_25, pred_agg.S_q_75, alpha=0.0, color=(custom_colors[1]-RGB(0.3,0.3,0.3),0.5))

    b32 = band!(ax32, pred_agg.t, pred_agg.E_q_p5, pred_agg.E_q_99p5, alpha=0.01, color=(custom_colors[2],0.4)) 
    l32 = lines!(ax32, pred_agg.t, pred_agg.E_mean, color=(custom_colors[2],1.0), linewidth=3)
    gtb32 = band!(ax32, pred_agg.t, pred_agg.E_q_10, pred_agg.E_q_90, alpha=0.0, color=(custom_colors[2]-RGB(0.1,0.1,0.1),0.5))
    gtb32_2 = band!(ax32, pred_agg.t, pred_agg.E_q_25, pred_agg.E_q_75, alpha=0.0, color=(custom_colors[2]-RGB(0.3,0.3,0.3),0.5))

    b33 = band!(ax33, pred_agg.t, pred_agg.I_q_p5, pred_agg.I_q_99p5, alpha=0.01, color=(custom_colors[3],0.4)) 
    l33 = lines!(ax33, pred_agg.t, pred_agg.I_mean, color=(custom_colors[3],1.0), linewidth=3)
    s33 = CairoMakie.scatter!(ax33, t_sim, X_sim[3,:], color = (custom_colors[3]-RGB(0.3,0.3,0.3),1), markersize=5)
    gtb33 = band!(ax33, pred_agg.t, pred_agg.I_q_10, pred_agg.I_q_90, alpha=0.0, color=(custom_colors[3]-RGB(0.1,0.1,0.1),0.5))
    gtb33_2 = band!(ax33, pred_agg.t, pred_agg.I_q_25, pred_agg.I_q_75, alpha=0.0, color=(custom_colors[3]-RGB(0.3,0.3,0.3),0.5))

    b34 = band!(ax34, pred_agg.t, pred_agg.R_q_p5, pred_agg.R_q_99p5, alpha=0.01, color=(custom_colors[4],0.4)) 
    s34 = CairoMakie.scatter!(ax34, t_sim, X_sim[4,:], color = (custom_colors[4]-RGB(0.3,0.3,0.3),1), markersize=5)
    l34 = lines!(ax34, pred_agg.t, pred_agg.R_mean, color=(custom_colors[4],1.0), linewidth=3)
    gtb34 = band!(ax34, pred_agg.t, pred_agg.R_q_10, pred_agg.R_q_90, alpha=0.0, color=(custom_colors[4]-RGB(0.1,0.1,0.1),0.5))
    gtb34_2 = band!(ax34, pred_agg.t, pred_agg.R_q_25, pred_agg.R_q_75, alpha=0.0, color=(custom_colors[4]-RGB(0.3,0.3,0.3),0.5))

    ############# line 3
    l41 = lines!(ax41, pred_agg.t, pred_agg.S_mean, color=(custom_colors[1],1.0), linewidth=3)
    r41 = lines!(ax41, ref.t, Array(ref)[1,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)
    for model_id in best_models
        df_trajectory_id = df_trajectory[df_trajectory.model_id.==Int(model_id), :]
        global l41s1 = lines!(ax41, df_trajectory_id.t, df_trajectory_id.S, color=custom_colors[1]-RGB(0.3,0.3,0.3), linewidth=1)
    end

    l42 = lines!(ax42, pred_agg.t, pred_agg.E_mean, color=(custom_colors[2],1.0), linewidth=3)
    r42 = lines!(ax42, ref.t, Array(ref)[2,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)
    for model_id in best_models
        df_trajectory_id = df_trajectory[df_trajectory.model_id.==Int(model_id), :]
        global l42s2 = lines!(ax42, df_trajectory_id.t, df_trajectory_id.E, color=custom_colors[2]-RGB(0.3,0.3,0.3), linewidth=1)
    end

    l43 = lines!(ax43, pred_agg.t, pred_agg.I_mean, color=(custom_colors[3],1.0), linewidth=3)
    r43 = lines!(ax43, ref.t, Array(ref)[3,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)
    s43 = CairoMakie.scatter!(ax43, t_sim, X_sim[3,:], color = (custom_colors[3]-RGB(0.3,0.3,0.3),1), markersize=5)
    for model_id in best_models
        df_trajectory_id = df_trajectory[df_trajectory.model_id.==Int(model_id), :]
        global l43s3 = lines!(ax43, df_trajectory_id.t, df_trajectory_id.I, color=custom_colors[3]-RGB(0.3,0.3,0.3), linewidth=1)
    end

    s44 = CairoMakie.scatter!(ax44, t_sim, X_sim[4,:], color = (custom_colors[4]-RGB(0.3,0.3,0.3),1), markersize=5)
    r44 = lines!(ax44, ref.t, Array(ref)[4,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)
    l44 = lines!(ax44, pred_agg.t, pred_agg.R_mean, color=(custom_colors[4],1.0), linewidth=3)
    for model_id in best_models
        df_trajectory_id = df_trajectory[df_trajectory.model_id.==Int(model_id), :]
        global l44s4 = lines!(ax44, df_trajectory_id.t, df_trajectory_id.R, color=custom_colors[4]-RGB(0.3,0.3,0.3), linewidth=1) 
    end

    legend45 = Legend(ga[3, 6],
        [[l41s1, l42s2, l43s3, l44s4]], # [hist, vline],
        ["10 best ensemble members"], orientation = :vertical, framevisible=false)


    legend25 = Legend(ga[2, 6],
        [r11, s_legend, l_legend, b_legend], # [hist, vline],
        ["ground truth", "observed", "posterior mean", "99% prediction interval"], orientation = :vertical, framevisible=false)


    resize_to_layout!(f)

    f
end

save(joinpath("paper_figures", "plots", "$(problem_name)_gaussian_0.01_overview_wo_reference.pdf"), trajectory_summary)


#========================================================================================================
    Plot overview of parameter estimates
========================================================================================================#

function get_cutoff(df, p_chi)
    return quantile(Distributions.Chisq(1),1-p_chi)/2 + minimum(skipmissing(df.negLL_obs_p_opt))
end

df_sub = df_stats[df_stats.negLL_obs_p_opt .<= get_cutoff(df_stats,0.05), :]
# select 99% best ensembles
df_sub = sort(df_sub, :negLL_obs_p_opt)[1:Int(floor(nrow(df_sub)*0.99)),:]

constant_parameters_summary = let
    f = Figure(size = (1200, 300),  px_per_unit = 10)
    ga = f[1, 1] = GridLayout()

    # Distribution of alpha
    ax1 = CairoMakie.Axis(ga[1, 1], title = "α", backgroundcolor=(:white), xlabelsize=20, titlesize=20)
    dens1 = CairoMakie.density!(ax1, df_sub.α_opt, normalization = :pdf, 
            label_size = 18, strokecolor = (:white, 0.5), color=(:steelblue4)) 
    vline = vlines!(ax1, [p_ode[1]]; color=:tomato4, linewidth=3)
    CairoMakie.ylims!(low=0)
    ax2 = CairoMakie.Axis(ga[1, 2], title = "γ", backgroundcolor=(:white), xlabelsize=20, titlesize=20)
    dens2 = CairoMakie.density!(ax2, df_sub.γ_opt, normalization = :pdf, 
            label_size = 18, strokecolor = (:white, 0.5), color=(:darkslategray4)) 
    vline = vlines!(ax2, [p_ode[2]]; color=:tomato4, linewidth=3)
    CairoMakie.ylims!(low=0)
    if noise_model=="Gaussian"
        global par_name = "σ"
    elseif noise_model=="negBin"
        global par_name = "overdispersion parameter"
    end
    ax3 = CairoMakie.Axis(ga[1, 3], title = par_name, backgroundcolor=(:white), xlabelsize=20, titlesize=20)
    dens3 = CairoMakie.density!(ax3, df_sub.np_opt, normalization = :pdf, 
            label_size = 18, strokecolor = (:white, 0.5), color=(:lightblue)) 
    vline = vlines!(ax3, [noise_par]; color=:tomato4, linewidth=3)
    CairoMakie.ylims!(low=0)

    Legend(ga[1, 4],
        [ vline],
        [ "ground truth"], framevisible=false, labelsize=23)
    f
end
save(joinpath("paper_figures", "plots", "$(problem_name)_gaussian_0.01_constant_parameters.pdf"), constant_parameters_summary)


#========================================================================================================
    Plot overview of beta
========================================================================================================#

time_varying_parameter_summary = let 
    f = Figure(size = (1200, 300),  px_per_unit = 10)

    ga = f[1, 1] = GridLayout()

    # Define axis of first line
    ax11 = CairoMakie.Axis(ga[1, 1], title = "A", xlabel="time", backgroundcolor=(:white), xlabelsize=20, ylabel="β", width=260, height=200, ylabelsize=22, titlesize=20, yticklabelsize=18, xticklabelsize=20) #, width=200)
    # ax12 = CairoMakie.Axis(ga[1, 2], title = "B", backgroundcolor=(:white), yticklabelsvisible=false, width=200, height=200) #, width=200)      
    ax13 = CairoMakie.Axis(ga[1, 2], title = "B", xlabel="time", backgroundcolor=(:white), yticklabelsvisible=false, width=260, height=200, xlabelsize=20, titlesize=20, xticklabelsize=20) #, width=200)      
    # ax14 = CairoMakie.Axis(ga[1, 3], title = "c", backgroundcolor=(:white), yticklabelsvisible=false, width=200, height=200) #, width=200)

    linkyaxes!(ax11,  ax13)

    # plot 1
    b11 = band!(ax11, pred_agg.t, pred_agg.β_q_p5, pred_agg.β_q_99p5, alpha=0.0, color=(palette(:default)[5],0.5))
    l11 = lines!(ax11, pred_agg.t, pred_agg.β_mean, color=(palette(:default)[5],1.0), linewidth=3)
    r11 = lines!(ax11, ref.t, β.(ref.t), color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)
    
    ####### line 3
    b31 = band!(ax13, pred_agg.t, pred_agg.β_q_p5, pred_agg.β_q_99p5, alpha=0.0, color=(palette(:default)[5],0.5))
    # l31 = lines!(ax13, pred_agg.t, pred_agg.β_mean, color=(palette(:default)[5],1.0), linewidth=3)
    gtb34 = band!(ax13, pred_agg.t, pred_agg.β_q_10, pred_agg.β_q_90, alpha=0.0, color=(palette(:default)[5]-RGB(0.1,0.1,0.1),0.5))
    gtb34_2 = band!(ax13, pred_agg.t, pred_agg.β_q_25, pred_agg.β_q_75, alpha=0.0, color=(palette(:default)[5]-RGB(0.3,0.3,0.3),0.5))

    f

    legend51 = Legend(ga[1,3], [[r11], [l11], [b11], [gtb34, gtb34_2]],
        ["ground truth", "posterior mean", "99% prediction interval", "80%/50% prediction interval"], framevisible=false, labelsize=20)

    resize_to_layout!(f)

    f
end

save(joinpath("paper_figures", "plots", "$(problem_name)_gaussian_0.01_time_varying_parameter_overview.pdf"), time_varying_parameter_summary)


#========================================================================================================
    Waterfall plots
========================================================================================================#
using ColorSchemes

fig = let 
    f = Figure(size = (1200, 300),  px_per_unit = 10)

    ga = f[1, 1] = GridLayout()

    # Define axis of first line
    ax11 = CairoMakie.Axis(ga[1, 1], title="all models",ylabel = "shifted NegLL", backgroundcolor=(:white), xlabel = "index", ylabelsize=22, titlesize=20, yticklabelsize=18, xticklabelsize=20, xlabelsize=20) #, width=200)
    ax12 = CairoMakie.Axis(ga[1, 2], title="8500 best", backgroundcolor=(:white), xlabel = "index", ylabelsize=22, titlesize=20, yticklabelsize=18, xticklabelsize=20, xlabelsize=20)
    ax13 = CairoMakie.Axis(ga[1, 3], title="300 best", backgroundcolor=(:white), xlabel = "index", ylabelsize=22, titlesize=20, yticklabelsize=18, xticklabelsize=20, xlabelsize=20)
   
    f

    # linkyaxes!(ax11, ax12)
    cmap = ColorSchemes.diverging_bky_60_10_c30_n256
    # Define content of first line
    y_vals = sort(df_stats, :negLL_obs_p_opt)[:, :negLL_obs_p_opt]
    cutoff_val = get_cutoff(df_stats, 0.05) .- minimum(y_vals)
    y_vals = y_vals .- minimum(y_vals)
    crange = (minimum(y_vals), maximum(y_vals)*0.01)

    l = lines!(ax11, 1:length(y_vals), y_vals, linewidth=1, color=:black)
    s = CairoMakie.scatter!(ax11, 1:length(y_vals), y_vals; color=y_vals, markersize=5, strokewidth=0, colorrange = crange, colormap =cmap)
    hline = hlines!(ax11, [cutoff_val], color=custom_colors[4], linewidth=3, linestyle=:dash)
    l_zoomed = lines!(ax12, 1:8500, y_vals[1:8500], linewidth=1, color=:black)
    s_zoomed = CairoMakie.scatter!(ax12, 1:8500, y_vals[1:8500]; color=y_vals[1:8500], markersize=5, strokewidth=0, colorrange = crange, colormap=cmap)
    hline2 = hlines!(ax12, [cutoff_val], color=custom_colors[4], linewidth=3, linestyle=:dash)
    l_zoomed2 = lines!(ax13, 1:300, y_vals[1:300], linewidth=1, color=:black)
    s_zoomed2 = CairoMakie.scatter!(ax13, 1:300, y_vals[1:300]; color=y_vals[1:300], markersize=5, strokewidth=0, colorrange = crange, colormap=cmap)
    f

    legend25 = Legend(ga[1, 4],
        [hline], # [hist, vline],
        ["cutoff value"], orientation = :vertical, framevisible=false, labelsize=20)

    f
end
save(joinpath("paper_figures", "plots", "$(problem_name)_waterfall.pdf"), fig)


#========================================================================================================
    Ensemble based for the different problems and noise models:
        SEIR multistart Gaussian 0.01
        SEIR multistart Gaussian 0.05 
        SEIR multistart NegBin 1.2
        SEIR multistart NegBin 2.5
========================================================================================================#

problem_name = "seir"
include("../experiments/seir.jl")
seir_01 = CSV.read(joinpath(joinpath(@__DIR__, "..", "experiments", "ensemble", "ensemble_10000", "seir", "multistart_gaussian_IR_noise_0.01"), "aggregated_predictions.csv"), DataFrame)
seir_05 = CSV.read(joinpath(joinpath(@__DIR__, "..", "experiments", "ensemble", "ensemble_10000", "seir", "multistart_gaussian_IR_noise_0.05"), "aggregated_predictions.csv"), DataFrame)
seir_12 = CSV.read(joinpath(joinpath(@__DIR__, "..", "experiments", "ensemble", "ensemble_10000", "seir", "multistart_negbin_IR_1.2"), "aggregated_predictions.csv"), DataFrame)
seir_25 = CSV.read(joinpath(joinpath(@__DIR__, "..", "experiments", "ensemble", "ensemble_10000", "seir", "multistart_negbin_IR_2.5"), "aggregated_predictions.csv"), DataFrame)

p_ode = [0.33, 0.05, 1.]
simulation_prob = ODEProblem(simulation_dynamics!, [p_ode[end]*0.995, p_ode[end]*0.004, p_ode[end]*0.001, 0.0], tspan, [0.33, 0.05, 1.])
t_sim_01, X_sim_01 = simulate_observations(n_timepoints, "Gaussian", 0.01, 1, 1, ode_prob=simulation_prob)
ref_01 = solve(simulation_prob, saveat = saveat_plot)

t_sim_05, X_sim_05 = simulate_observations(n_timepoints, "Gaussian", 0.05, 1, 1, ode_prob=simulation_prob)
ref_05 = ref_01

p_ode = [0.33, 0.05, 1000.]
simulation_prob = ODEProblem(simulation_dynamics!, [p_ode[end]*0.995, p_ode[end]*0.004, p_ode[end]*0.001, 0.0], tspan, [0.33, 0.05, 1000.])
t_sim_12, X_sim_12 = simulate_observations(n_timepoints, "negBin", 1.2, 1, 1, ode_prob=simulation_prob)
ref_12 = solve(simulation_prob, saveat = saveat_plot)

t_sim_25, X_sim_25 = simulate_observations(n_timepoints, "negBin", 2.5, 1, 1, ode_prob=simulation_prob)
ref_25 = ref_12

custom_colors = palette(:tab10)[1:4] # get(ColorSchemes.viridis, [i/(n_colors-1) for i in 0:1:(n_colors-1)])

noise_model_comparison = let 
    f = Figure(size = (1700, 650),  px_per_unit = 10)

    ga = f[1, 1] = GridLayout()

    # Define axis of first line
    ax11 = CairoMakie.Axis(ga[1, 2], title = "S", backgroundcolor=(:white), xlabelsize=20, 
        xticklabelsvisible=false) #, width=200)
    ax12 = CairoMakie.Axis(ga[1, 3], title = "E", backgroundcolor=(:white), xticklabelsvisible=false, yticklabelsvisible=false) #, width=200)      
    ax13 = CairoMakie.Axis(ga[1, 4], title = "I", backgroundcolor=(:white), xticklabelsvisible=false, yticklabelsvisible=false) #, width=200)      
    ax14 = CairoMakie.Axis(ga[1, 5], title = "R", backgroundcolor=(:white), xticklabelsvisible=false, yticklabelsvisible=false) #, width=200)

    ax1 = CairoMakie.Axis(ga[1,1], width = 80)
    hidedecorations!(ax1)  # hides ticks, grid and lables
    hidespines!(ax1)
    text!(ax1, "Gaussian \n(σ = 0.01)", fontsize=15,space = :relative, font=:bold, position=Point2f(0.,0.6))
    ax2 = CairoMakie.Axis(ga[2,1], width = 80)
    hidedecorations!(ax2)  # hides ticks, grid and lables
    hidespines!(ax2)
    text!(ax2, "Gaussian \n(σ = 0.05)", fontsize=15,space = :relative, font=:bold, position=Point2f(0.,0.6))
    ax3 = CairoMakie.Axis(ga[3,1], width = 80)
    hidedecorations!(ax3)  # hides ticks, grid and lables
    hidespines!(ax3)
    text!(ax3, "NegBin \n(d = 1.2)", fontsize=15,space = :relative, font=:bold, position=Point2f(0.,0.6))
    ax4 = CairoMakie.Axis(ga[4,1], width = 80)
    hidedecorations!(ax4)  # hides ticks, grid and lables
    hidespines!(ax4)
    text!(ax4, "NegBin \n(d = 2.5)", fontsize=15,space = :relative, font=:bold, position=Point2f(0.,0.6))
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

    b_legend = band!(ax11, seir_01.t, seir_01.S_q_p5, seir_01.S_q_99p5, alpha=0.0, color=RGB(0.7,0.7,0.7))
    l_legend = lines!(ax11, seir_01.t, seir_01.S_mean, color=RGB(0.4, 0.4, 0.4), linewidth=3)
    s_legend = CairoMakie.scatter!(ax13, t_sim_01, X_sim_01[3,:], color=RGB(0.4, 0.4, 0.4), markersize=7)

    # Define content of first line
    b11 = band!(ax11, seir_01.t, seir_01.S_q_p5, seir_01.S_q_99p5, alpha=0.0, color=custom_bands[1])
    l11 = lines!(ax11, seir_01.t, seir_01.S_mean, color=(custom_colors[1],1.0), linewidth=3)
    r11 = lines!(ax11, ref_01.t, Array(ref_01)[1,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

    b12 = band!(ax12, seir_01.t, seir_01.E_q_p5, seir_01.E_q_99p5, alpha=0.01, color=(custom_colors[2],0.4)) 
    l12 = lines!(ax12, seir_01.t, seir_01.E_mean, color=(custom_colors[2],1.0), linewidth=3)
    r12 = lines!(ax12, ref_01.t, Array(ref_01)[2,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

    b13 = band!(ax13, seir_01.t, seir_01.I_q_p5, seir_01.I_q_99p5, alpha=0.01, color=(custom_colors[3],0.4)) 
    l13 = lines!(ax13, seir_01.t, seir_01.I_mean, color=(custom_colors[3],1.0), linewidth=3)
    s13 = CairoMakie.scatter!(ax13, t_sim_01, X_sim_01[3,:], color = (custom_colors[3]-RGB(0.3,0.3,0.3),1), markersize=5)
    r13 = lines!(ax13, ref_01.t, Array(ref_01)[3,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

    b14 = band!(ax14, seir_01.t, seir_01.R_q_p5, seir_01.R_q_99p5, alpha=0.01, color=(custom_colors[4],0.4)) 
    l14 = lines!(ax14, seir_01.t, seir_01.R_mean, color=(custom_colors[4],1.0), linewidth=3)
    s14 = CairoMakie.scatter!(ax14, t_sim_01, X_sim_01[4,:], color = (custom_colors[4]-RGB(0.3,0.3,0.3),1), markersize=5)
    r14 = lines!(ax14, ref_01.t, Array(ref_01)[4,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

    # Define content of second line
    b21 = band!(ax21, seir_05.t, seir_05.S_q_p5, seir_05.S_q_99p5, alpha=0.0, color=(custom_colors[1],0.4))
    l21 = lines!(ax21, seir_05.t, seir_05.S_mean, color=(custom_colors[1],1.0), linewidth=3)
    r21 = lines!(ax21, ref_05.t, Array(ref_05)[1,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

    b22 = band!(ax22, seir_05.t, seir_05.E_q_p5, seir_05.E_q_99p5, alpha=0.01, color=(custom_colors[2],0.4)) 
    l22 = lines!(ax22, seir_05.t, seir_05.E_mean, color=(custom_colors[2],1.0), linewidth=3)
    r22 = lines!(ax22, ref_05.t, Array(ref_05)[2,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

    b23 = band!(ax23, seir_05.t, seir_05.I_q_p5, seir_05.I_q_99p5, alpha=0.01, color=(custom_colors[3],0.4)) 
    l23 = lines!(ax23, seir_05.t, seir_05.I_mean, color=(custom_colors[3],1.0), linewidth=3)
    s23 = CairoMakie.scatter!(ax23, t_sim_05, X_sim_05[3,:], color = (custom_colors[3]-RGB(0.3,0.3,0.3),1), markersize=5)
    r23 = lines!(ax23, ref_05.t, Array(ref_05)[3,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

    b24 = band!(ax24, seir_05.t, seir_05.R_q_p5, seir_05.R_q_99p5, alpha=0.01, color=(custom_colors[4],0.4)) 
    l24 = lines!(ax24, seir_05.t, seir_05.R_mean, color=(custom_colors[4],1.0), linewidth=3)
    s24 = CairoMakie.scatter!(ax24, t_sim_05, X_sim_05[4,:], color = (custom_colors[4]-RGB(0.3,0.3,0.3),1), markersize=5)
    r24 = lines!(ax24, ref_05.t, Array(ref_05)[4,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

    ####### line 3
    b31 = band!(ax31, seir_12.t, seir_12.S_q_p5, seir_12.S_q_99p5, alpha=0.0, color=(custom_colors[1],0.4))
    l31 = lines!(ax31, seir_12.t, seir_12.S_mean, color=(custom_colors[1],1.0), linewidth=3)
    r31 = lines!(ax31, ref_12.t, Array(ref_12)[1,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

    b32 = band!(ax32, seir_12.t, seir_12.E_q_p5, seir_12.E_q_99p5, alpha=0.01, color=(custom_colors[2],0.4)) 
    l32 = lines!(ax32, seir_12.t, seir_12.E_mean, color=(custom_colors[2],1.0), linewidth=3)
    r32 = lines!(ax32, ref_12.t, Array(ref_12)[2,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

    b33 = band!(ax33, seir_12.t, seir_12.I_q_p5, seir_12.I_q_99p5, alpha=0.01, color=(custom_colors[3],0.4)) 
    l33 = lines!(ax33, seir_12.t, seir_12.I_mean, color=(custom_colors[3],1.0), linewidth=3)
    s33 = CairoMakie.scatter!(ax33, t_sim_12, X_sim_12[3,:], color = (custom_colors[3]-RGB(0.3,0.3,0.3),1), markersize=5)
    r33 = lines!(ax33, ref_12.t, Array(ref_12)[3,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

    b34 = band!(ax34, seir_12.t, seir_12.R_q_p5, seir_12.R_q_99p5, alpha=0.01, color=(custom_colors[4],0.4)) 
    l34 = lines!(ax34, seir_12.t, seir_12.R_mean, color=(custom_colors[4],1.0), linewidth=3)
    s34 = CairoMakie.scatter!(ax34, t_sim_12, X_sim_12[4,:], color = (custom_colors[4]-RGB(0.3,0.3,0.3),1), markersize=5)
    r34 = lines!(ax34, ref_12.t, Array(ref_12)[4,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)


    ############# line 4
    # Define content of second line
    b41 = band!(ax41, seir_25.t, seir_25.S_q_p5, seir_25.S_q_99p5, alpha=0.0, color=(custom_colors[1],0.4))
    l41 = lines!(ax41, seir_25.t, seir_25.S_mean, color=(custom_colors[1],1.0), linewidth=3)
    r41 = lines!(ax41, ref_25.t, Array(ref_25)[1,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

    b42 = band!(ax42, seir_25.t, seir_25.E_q_p5, seir_25.E_q_99p5, alpha=0.01, color=(custom_colors[2],0.4)) 
    l42 = lines!(ax42, seir_25.t, seir_25.E_mean, color=(custom_colors[2],1.0), linewidth=3)
    r42 = lines!(ax42, ref_25.t, Array(ref_25)[2,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

    b43 = band!(ax43, seir_25.t, seir_25.I_q_p5, seir_25.I_q_99p5, alpha=0.01, color=(custom_colors[3],0.4)) 
    l43 = lines!(ax43, seir_25.t, seir_25.I_mean, color=(custom_colors[3],1.0), linewidth=3)
    s43 = CairoMakie.scatter!(ax43, t_sim_25, X_sim_25[3,:], color = (custom_colors[3]-RGB(0.3,0.3,0.3),1), markersize=5)
    r43 = lines!(ax43, ref_25.t, Array(ref_25)[3,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

    b44 = band!(ax44, seir_25.t, seir_25.R_q_p5, seir_25.R_q_99p5, alpha=0.01, color=(custom_colors[4],0.4)) 
    l44 = lines!(ax44, seir_25.t, seir_25.R_mean, color=(custom_colors[4],1.0), linewidth=3)
    s44 = CairoMakie.scatter!(ax44, t_sim_25, X_sim_25[4,:], color = (custom_colors[4]-RGB(0.3,0.3,0.3),1), markersize=5)
    r44 = lines!(ax44, ref_25.t, Array(ref_25)[4,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)
    
    legend25 = Legend(ga[2:3, 6],
        [r11, s_legend, l_legend, b_legend], # [hist, vline],
        ["ground truth", "observed", "posterior mean", "99% prediction interval"], orientation = :vertical, framevisible=false)


    resize_to_layout!(f)

    f
end

save(joinpath("paper_figures", "plots", "seir_noise_model_comparison.pdf"), noise_model_comparison)


# same for seir_pulse
problem_name = "seir_pulse"
include("../experiments/seir_pulse.jl")

seir_01 = CSV.read(joinpath(joinpath(@__DIR__, "..", "experiments", "ensemble", "ensemble_10000", "seir_pulse", "multistart_gaussian_IR_noise_0.01"), "aggregated_predictions.csv"), DataFrame)
seir_05 = CSV.read(joinpath(joinpath(@__DIR__, "..", "experiments", "ensemble", "ensemble_10000", "seir_pulse", "multistart_gaussian_IR_noise_0.03"), "aggregated_predictions.csv"), DataFrame)
seir_12 = CSV.read(joinpath(joinpath(@__DIR__, "..", "experiments", "ensemble", "ensemble_10000", "seir_pulse", "multistart_negbin_IR_1.2"), "aggregated_predictions.csv"), DataFrame)
seir_25 = CSV.read(joinpath(joinpath(@__DIR__, "..", "experiments", "ensemble", "ensemble_10000", "seir_pulse", "multistart_negbin_IR_2.5"), "aggregated_predictions.csv"), DataFrame)

p_ode = [0.9, 0.1, 1.] 
simulation_prob = ODEProblem(simulation_dynamics!, [p_ode[end]*0.995, p_ode[end]*0.004, p_ode[end]*0.001, 0.0], tspan, p_ode)
t_sim_01, X_sim_01 = simulate_observations(n_timepoints, "Gaussian", 0.01, 1, 1, ode_prob=simulation_prob)
ref_01 = solve(simulation_prob, saveat = saveat_plot)

t_sim_05, X_sim_05 = simulate_observations(n_timepoints, "Gaussian", 0.05, 1, 1, ode_prob=simulation_prob)
ref_05 = ref_01

p_ode = [0.9, 0.1, 1000.] 
simulation_prob = ODEProblem(simulation_dynamics!, [p_ode[end]*0.995, p_ode[end]*0.004, p_ode[end]*0.001, 0.0], tspan, p_ode)
t_sim_12, X_sim_12 = simulate_observations(n_timepoints, "negBin", 1.2, 1, 1, ode_prob=simulation_prob)
ref_12 = solve(simulation_prob, saveat = saveat_plot)

t_sim_25, X_sim_25 = simulate_observations(n_timepoints, "negBin", 2.5, 1, 1, ode_prob=simulation_prob)
ref_25 = ref_12

noise_model_comparison = let 
    f = Figure(size = (1700, 650),  px_per_unit = 10)

    ga = f[1, 1] = GridLayout()

    # Define axis of first line
    ax11 = CairoMakie.Axis(ga[1, 2], title = "S", backgroundcolor=(:white), xlabelsize=20, 
        xticklabelsvisible=false) #, width=200)
    ax12 = CairoMakie.Axis(ga[1, 3], title = "E", backgroundcolor=(:white), xticklabelsvisible=false, yticklabelsvisible=false) #, width=200)      
    ax13 = CairoMakie.Axis(ga[1, 4], title = "I", backgroundcolor=(:white), xticklabelsvisible=false, yticklabelsvisible=false) #, width=200)      
    ax14 = CairoMakie.Axis(ga[1, 5], title = "R", backgroundcolor=(:white), xticklabelsvisible=false, yticklabelsvisible=false) #, width=200)

    ax1 = CairoMakie.Axis(ga[1,1], width = 80)
    hidedecorations!(ax1)  # hides ticks, grid and lables
    hidespines!(ax1)
    text!(ax1, "Gaussian \n(σ = 0.01)", fontsize=15,space = :relative, font=:bold, position=Point2f(0.,0.6))
    ax2 = CairoMakie.Axis(ga[2,1], width = 80)
    hidedecorations!(ax2)  # hides ticks, grid and lables
    hidespines!(ax2)
    text!(ax2, "Gaussian \n(σ = 0.05)", fontsize=15,space = :relative, font=:bold, position=Point2f(0.,0.6))
    ax3 = CairoMakie.Axis(ga[3,1], width = 80)
    hidedecorations!(ax3)  # hides ticks, grid and lables
    hidespines!(ax3)
    text!(ax3, "NegBin \n(d = 1.2)", fontsize=15,space = :relative, font=:bold, position=Point2f(0.,0.6))
    ax4 = CairoMakie.Axis(ga[4,1], width = 80)
    hidedecorations!(ax4)  # hides ticks, grid and lables
    hidespines!(ax4)
    text!(ax4, "NegBin \n(d = 2.5)", fontsize=15,space = :relative, font=:bold, position=Point2f(0.,0.6))
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

    b_legend = band!(ax11, seir_01.t, seir_01.S_q_p5, seir_01.S_q_99p5, alpha=0.0, color=RGB(0.7,0.7,0.7))
    l_legend = lines!(ax11, seir_01.t, seir_01.S_mean, color=RGB(0.4, 0.4, 0.4), linewidth=3)
    s_legend = CairoMakie.scatter!(ax13, t_sim_01, X_sim_01[3,:], color=RGB(0.4, 0.4, 0.4), markersize=7)

    # Define content of first line
    b11 = band!(ax11, seir_01.t, seir_01.S_q_p5, seir_01.S_q_99p5, alpha=0.0, color=custom_bands[1])
    l11 = lines!(ax11, seir_01.t, seir_01.S_mean, color=(custom_colors[1],1.0), linewidth=3)
    r11 = lines!(ax11, ref_01.t, Array(ref_01)[1,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

    b12 = band!(ax12, seir_01.t, seir_01.E_q_p5, seir_01.E_q_99p5, alpha=0.01, color=(custom_colors[2],0.4)) 
    l12 = lines!(ax12, seir_01.t, seir_01.E_mean, color=(custom_colors[2],1.0), linewidth=3)
    r12 = lines!(ax12, ref_01.t, Array(ref_01)[2,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

    b13 = band!(ax13, seir_01.t, seir_01.I_q_p5, seir_01.I_q_99p5, alpha=0.01, color=(custom_colors[3],0.4)) 
    l13 = lines!(ax13, seir_01.t, seir_01.I_mean, color=(custom_colors[3],1.0), linewidth=3)
    s13 = CairoMakie.scatter!(ax13, t_sim_01, X_sim_01[3,:], color = (custom_colors[3]-RGB(0.3,0.3,0.3),1), markersize=5)
    r13 = lines!(ax13, ref_01.t, Array(ref_01)[3,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

    b14 = band!(ax14, seir_01.t, seir_01.R_q_p5, seir_01.R_q_99p5, alpha=0.01, color=(custom_colors[4],0.4)) 
    l14 = lines!(ax14, seir_01.t, seir_01.R_mean, color=(custom_colors[4],1.0), linewidth=3)
    s14 = CairoMakie.scatter!(ax14, t_sim_01, X_sim_01[4,:], color = (custom_colors[4]-RGB(0.3,0.3,0.3),1), markersize=5)
    r14 = lines!(ax14, ref_01.t, Array(ref_01)[4,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

    # Define content of second line
    b21 = band!(ax21, seir_05.t, seir_05.S_q_p5, seir_05.S_q_99p5, alpha=0.0, color=(custom_colors[1],0.4))
    l21 = lines!(ax21, seir_05.t, seir_05.S_mean, color=(custom_colors[1],1.0), linewidth=3)
    r21 = lines!(ax21, ref_05.t, Array(ref_05)[1,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

    b22 = band!(ax22, seir_05.t, seir_05.E_q_p5, seir_05.E_q_99p5, alpha=0.01, color=(custom_colors[2],0.4)) 
    l22 = lines!(ax22, seir_05.t, seir_05.E_mean, color=(custom_colors[2],1.0), linewidth=3)
    r22 = lines!(ax22, ref_05.t, Array(ref_05)[2,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

    b23 = band!(ax23, seir_05.t, seir_05.I_q_p5, seir_05.I_q_99p5, alpha=0.01, color=(custom_colors[3],0.4)) 
    l23 = lines!(ax23, seir_05.t, seir_05.I_mean, color=(custom_colors[3],1.0), linewidth=3)
    s23 = CairoMakie.scatter!(ax23, t_sim_05, X_sim_05[3,:], color = (custom_colors[3]-RGB(0.3,0.3,0.3),1), markersize=5)
    r23 = lines!(ax23, ref_05.t, Array(ref_05)[3,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

    b24 = band!(ax24, seir_05.t, seir_05.R_q_p5, seir_05.R_q_99p5, alpha=0.01, color=(custom_colors[4],0.4)) 
    l24 = lines!(ax24, seir_05.t, seir_05.R_mean, color=(custom_colors[4],1.0), linewidth=3)
    s24 = CairoMakie.scatter!(ax24, t_sim_05, X_sim_05[4,:], color = (custom_colors[4]-RGB(0.3,0.3,0.3),1), markersize=5)
    r24 = lines!(ax24, ref_05.t, Array(ref_05)[4,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

    ####### line 3
    b31 = band!(ax31, seir_12.t, seir_12.S_q_p5, seir_12.S_q_99p5, alpha=0.0, color=(custom_colors[1],0.4))
    l31 = lines!(ax31, seir_12.t, seir_12.S_mean, color=(custom_colors[1],1.0), linewidth=3)
    r31 = lines!(ax31, ref_12.t, Array(ref_12)[1,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

    b32 = band!(ax32, seir_12.t, seir_12.E_q_p5, seir_12.E_q_99p5, alpha=0.01, color=(custom_colors[2],0.4)) 
    l32 = lines!(ax32, seir_12.t, seir_12.E_mean, color=(custom_colors[2],1.0), linewidth=3)
    r32 = lines!(ax32, ref_12.t, Array(ref_12)[2,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

    b33 = band!(ax33, seir_12.t, seir_12.I_q_p5, seir_12.I_q_99p5, alpha=0.01, color=(custom_colors[3],0.4)) 
    l33 = lines!(ax33, seir_12.t, seir_12.I_mean, color=(custom_colors[3],1.0), linewidth=3)
    s33 = CairoMakie.scatter!(ax33, t_sim_12, X_sim_12[3,:], color = (custom_colors[3]-RGB(0.3,0.3,0.3),1), markersize=5)
    r33 = lines!(ax33, ref_12.t, Array(ref_12)[3,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

    b34 = band!(ax34, seir_12.t, seir_12.R_q_p5, seir_12.R_q_99p5, alpha=0.01, color=(custom_colors[4],0.4)) 
    l34 = lines!(ax34, seir_12.t, seir_12.R_mean, color=(custom_colors[4],1.0), linewidth=3)
    s34 = CairoMakie.scatter!(ax34, t_sim_12, X_sim_12[4,:], color = (custom_colors[4]-RGB(0.3,0.3,0.3),1), markersize=5)
    r34 = lines!(ax34, ref_12.t, Array(ref_12)[4,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)


    ############# line 4
    # Define content of second line
    b41 = band!(ax41, seir_25.t, seir_25.S_q_p5, seir_25.S_q_99p5, alpha=0.0, color=(custom_colors[1],0.4))
    l41 = lines!(ax41, seir_25.t, seir_25.S_mean, color=(custom_colors[1],1.0), linewidth=3)
    r41 = lines!(ax41, ref_25.t, Array(ref_25)[1,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

    b42 = band!(ax42, seir_25.t, seir_25.E_q_p5, seir_25.E_q_99p5, alpha=0.01, color=(custom_colors[2],0.4)) 
    l42 = lines!(ax42, seir_25.t, seir_25.E_mean, color=(custom_colors[2],1.0), linewidth=3)
    r42 = lines!(ax42, ref_25.t, Array(ref_25)[2,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

    b43 = band!(ax43, seir_25.t, seir_25.I_q_p5, seir_25.I_q_99p5, alpha=0.01, color=(custom_colors[3],0.4)) 
    l43 = lines!(ax43, seir_25.t, seir_25.I_mean, color=(custom_colors[3],1.0), linewidth=3)
    s43 = CairoMakie.scatter!(ax43, t_sim_25, X_sim_25[3,:], color = (custom_colors[3]-RGB(0.3,0.3,0.3),1), markersize=5)
    r43 = lines!(ax43, ref_25.t, Array(ref_25)[3,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

    b44 = band!(ax44, seir_25.t, seir_25.R_q_p5, seir_25.R_q_99p5, alpha=0.01, color=(custom_colors[4],0.4)) 
    l44 = lines!(ax44, seir_25.t, seir_25.R_mean, color=(custom_colors[4],1.0), linewidth=3)
    s44 = CairoMakie.scatter!(ax44, t_sim_25, X_sim_25[4,:], color = (custom_colors[4]-RGB(0.3,0.3,0.3),1), markersize=5)
    r44 = lines!(ax44, ref_25.t, Array(ref_25)[4,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)
    
    legend25 = Legend(ga[2:3, 6],
        [r11, s_legend, l_legend, b_legend], # [hist, vline],
        ["ground truth", "observed", "posterior mean", "99% prediction interval"], orientation = :vertical, framevisible=false)


    resize_to_layout!(f)

    f
end


save(joinpath("paper_figures", "plots", "seir_pulse_noise_model_comparison.pdf"), noise_model_comparison)