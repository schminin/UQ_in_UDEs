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
problem_name = "quadratic_dynamics"
experiment_name = "multistart_gaussian_noise_0.01"

experiment_path = joinpath(@__DIR__, "..", "experiments", "ensemble", "ensemble_10000", problem_name, experiment_name)

pred_agg = CSV.read(joinpath(experiment_path, "aggregated_predictions.csv"), DataFrame)

ref = solve(ODEProblem(simulation_dynamics!, u0, tspan, p_ode), saveat = saveat_plot)

custom_colors = palette(:tab10)[1:5] 
fig = let 
    f = Figure(size = (1200, 300),  px_per_unit = 10)

    ga = f[1, 1] = GridLayout()

    # Define axis of first line
    ax11 = CairoMakie.Axis(ga[1, 1], title = "A", xlabel="time", backgroundcolor=(:white), xlabelsize=20, ylabel="x", width=260, height=200, ylabelsize=22, titlesize=20, yticklabelsize=18, xticklabelsize=20) #, width=200)
    # ax12 = CairoMakie.Axis(ga[1, 2], title = "B", backgroundcolor=(:white), yticklabelsvisible=false, width=200, height=200) #, width=200)      
    ax13 = CairoMakie.Axis(ga[1, 2], title = "B", xlabel="time", backgroundcolor=(:white), yticklabelsvisible=false, width=260, height=200, xlabelsize=20, titlesize=20, xticklabelsize=20) #, width=200)      
    # ax14 = CairoMakie.Axis(ga[1, 3], title = "c", backgroundcolor=(:white), yticklabelsvisible=false, width=200, height=200) #, width=200)
    s11 = CairoMakie.scatter!(ax11, t_sim, X_sim[1,:], color=RGB(0.3,0.3,0.3), markersize=7)

    linkyaxes!(ax11,  ax13)

    # plot 1
    b11 = band!(ax11, pred_agg.t, pred_agg.x_q_p5, pred_agg.x_q_99p5, alpha=0.0, color=(custom_colors[5],0.5))
    l11 = lines!(ax11, pred_agg.t, pred_agg.x_mean, color=(custom_colors[5],1.0), linewidth=3)
    r11 = lines!(ax11, ref.t, Array(ref)[1,:], color=RGB(0.3,0.3,0.3), linewidth=2, linestyle=:dash)

    ####### line 3
    b31 = band!(ax13, pred_agg.t, pred_agg.x_q_p5, pred_agg.x_q_99p5, alpha=0.0, color=(custom_colors[5],0.5))
    # l31 = lines!(ax13, pred_agg.t, pred_agg.x_mean, color=(custom_colors[5],1.0), linewidth=3)
    gtb34 = band!(ax13, pred_agg.t, pred_agg.x_q_10, pred_agg.x_q_90, alpha=0.0, color=(custom_colors[5]-RGB(0.1,0.1,0.1),0.5))
    gtb34_2 = band!(ax13, pred_agg.t, pred_agg.x_q_25, pred_agg.x_q_75, alpha=0.0, color=(custom_colors[5]-RGB(0.3,0.3,0.3),0.5))
    s11 = CairoMakie.scatter!(ax13, t_sim, X_sim[1,:], color=RGB(0.3,0.3,0.3), markersize=7)

    f

    legend51 = Legend(ga[1,3], [[r11], [s11], [l11], [b11], [gtb34, gtb34_2]],
        ["ground truth", "observations", "posterior mean", "99% prediction interval", "80%/50% prediction interval"], framevisible=false, labelsize=20)

    resize_to_layout!(f)

    f
end
save(joinpath("paper_figures", "plots", "$(problem_name)_gaussian_0.01.pdf"), fig)
