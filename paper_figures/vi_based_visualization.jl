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
using Pigeons

#========================================================================================================
    Experimental Settings
========================================================================================================#
problem_name = "seir_pulse"
experiment_name = "gaussian_IR_noise_0.03"

experiment_path = joinpath(@__DIR__, "..", "experiments", "variational_inference", "standard_nn", problem_name, experiment_name)
# ground_truth_path = joinpath(@__DIR__, "..", "experiments", "ensemble", "ensemble_ground_truth", problem_name, ground_truth)

df_trajectory = CSV.read(joinpath(experiment_path, "posterior_trajectory.csv"), DataFrame)
df_stats = CSV.read(joinpath(experiment_path, "posterior_stats.csv"), DataFrame)
pred_agg = CSV.read(joinpath(experiment_path, "aggregated_predictions.csv"), DataFrame)

q = load(joinpath(experiment_path, "posteriors", "advi_100000.jld2"))["q"]
sampled_data = rand(q, 10000)

# pred_agg_ground_truth = CSV.read(joinpath(ground_truth_path, "aggregated_predictions.csv"), DataFrame)

include(joinpath(experiment_path, "experimental_setting.jl")) 
include("../experiments/$(problem_name).jl")
include("../experiments/simulate.jl")
include("../experiments/utils.jl")
ref = solve(ODEProblem(simulation_dynamics!, u0, tspan, p_ode), saveat = saveat_plot)

custom_colors = palette(:tab10)[1:4] # get(ColorSchemes.viridis, [i/(n_colors-1) for i in 0:1:(n_colors-1)])
custom_bands = [RGB(165/255, 200/255, 225/255), RGB(255/255, 203/255, 158/255), RGB(170/255, 217/255, 170/255), RGB(238/255, 168/255, 169/255)]

CairoMakie.activate!(type = "pdf")

#========================================================================================================
    Plot overview of parameter estimates for the SEIR problems
========================================================================================================#
mcmc_res = Chains(pt)

constant_parameters_summary = let
    f = Figure(size = (1200, 300),  px_per_unit = 10)
    ga = f[1, 1] = GridLayout()

    # Distribution of alpha
    ax1 = CairoMakie.Axis(ga[1, 1], title = "α", backgroundcolor=(:white), xlabelsize=20, titlesize=20)
    dens1 = CairoMakie.density!(ax1, retrieve_α.(sampled_data[1,:]), normalization = :pdf, 
            label_size = 18, strokecolor = (:white, 0.5), color=(:steelblue4)) 
    vline = vlines!(ax1, [p_ode[1]]; color=:tomato4, linewidth=3)
    CairoMakie.ylims!(low=0)
    ax2 = CairoMakie.Axis(ga[1, 2], title = "γ", backgroundcolor=(:white), xlabelsize=20, titlesize=20)
    dens2 = CairoMakie.density!(ax2, retrieve_γ.(sampled_data[2,:]), normalization = :pdf, 
            label_size = 18, strokecolor = (:white, 0.5), color=(:darkslategray4)) 
    vline = vlines!(ax2, [p_ode[2]]; color=:tomato4, linewidth=3)
    CairoMakie.ylims!(low=0)
    if noise_model=="Gaussian"
        global par_name = "σ"
    elseif noise_model=="negBin"
        global par_name = "overdispersion parameter"
    end
    ax3 = CairoMakie.Axis(ga[1, 3], title = par_name, backgroundcolor=(:white), xlabelsize=20, titlesize=20)
    if noise_model=="Gaussian"
        dens3 = CairoMakie.density!(ax3, sqrt.(exp.(sampled_data[3,:])), normalization = :pdf, 
                    label_size = 18, strokecolor = (:white, 0.5), color=(:lightblue)) 
                    vline = vlines!(ax3, [noise_par]; color=:tomato4, linewidth=3)
    elseif noise_model=="negBin"
        dens3 = CairoMakie.density!(ax3, 1 .+ exp.((sampled_data[3,:])), normalization = :pdf, 
                    label_size = 18, strokecolor = (:white, 0.5), color=(:lightblue)) 
                    vline = vlines!(ax3, [noise_par]; color=:tomato4, linewidth=3)
    end

    CairoMakie.ylims!(low=0)

    Legend(ga[1, 4],
        [ vline],
        [ "ground truth"], framevisible=false, labelsize=23)
    f
end
save(joinpath("paper_figures", "plots", problem_name, experiment_name, "vi_constant_parameters.pdf"), constant_parameters_summary)


#========================================================================================================
    Plot overview of beta
========================================================================================================#

time_varying_parameter_summary = let 
    f = Figure(size = (1200, 300),  px_per_unit = 10)

    ga = f[1, 1] = GridLayout()

    # Define axis of first line
    ax11 = CairoMakie.Axis(ga[1, 1], xlabel="time", backgroundcolor=(:white), xlabelsize=20, ylabel="β", width=260, height=200, ylabelsize=22, titlesize=20, yticklabelsize=18, xticklabelsize=20) #, width=200)
    # ax12 = CairoMakie.Axis(ga[1, 2], title = "B", backgroundcolor=(:white), yticklabelsvisible=false, width=200, height=200) #, width=200)      
    # ax13 = CairoMakie.Axis(ga[1, 2], title = "B", xlabel="time", backgroundcolor=(:white), yticklabelsvisible=false, width=260, height=200, xlabelsize=20, titlesize=20, xticklabelsize=20) #, width=200)      
    # ax14 = CairoMakie.Axis(ga[1, 3], title = "c", backgroundcolor=(:white), yticklabelsvisible=false, width=200, height=200) #, width=200)

    #linkyaxes!(ax11,  ax13)

    # plot 1
    b11 = band!(ax11, pred_agg.t, pred_agg.β_q_p5, pred_agg.β_q_99p5, alpha=0.0, color=(palette(:default)[5],0.5))
    l11 = lines!(ax11, pred_agg.t, pred_agg.β_mean, color=(palette(:default)[5],1.0), linewidth=3)
    r11 = lines!(ax11, ref.t, β.(ref.t), color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)
    
    ####### line 3
    #b31 = band!(ax13, pred_agg.t, pred_agg.β_q_p5, pred_agg.β_q_99p5, alpha=0.0, color=(palette(:default)[5],0.5))
    # l31 = lines!(ax13, pred_agg.t, pred_agg.β_mean, color=(palette(:default)[5],1.0), linewidth=3)
    #gtb34 = band!(ax13, pred_agg.t, pred_agg.β_q_10, pred_agg.β_q_90, alpha=0.0, color=(palette(:default)[5]-RGB(0.1,0.1,0.1),0.5))
    #gtb34_2 = band!(ax13, pred_agg.t, pred_agg.β_q_25, pred_agg.β_q_75, alpha=0.0, color=(palette(:default)[5]-RGB(0.3,0.3,0.3),0.5))

    f

    legend51 = Legend(ga[1,2], [[r11], [l11], [b11]],
        ["ground truth", "posterior mean", "99% prediction interval"], framevisible=false, labelsize=20)

    resize_to_layout!(f)

    f
end

save(joinpath("paper_figures", "plots", problem_name, experiment_name, "vi_time_varying_parameter_overview.pdf"), time_varying_parameter_summary)

