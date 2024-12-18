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
experiment_name = "gaussian_IR_noise_0.05"

isdir(joinpath("paper_figures", "plots", "$(problem_name)")) || mkdir(joinpath("paper_figures", "plots", problem_name))
isdir(joinpath("paper_figures", "plots", "$(problem_name)", experiment_name)) || mkdir(joinpath("paper_figures", "plots", problem_name, experiment_name))

ensemble = CSV.read(joinpath(joinpath(@__DIR__, "..", "experiments", "ensemble", "ensemble_10000", problem_name, "multistart_"*experiment_name), "aggregated_predictions.csv"), DataFrame)
mcmc = CSV.read(joinpath(joinpath(@__DIR__, "..", "experiments", "MCMC", "pigeon", problem_name, experiment_name), "aggregate_predictions.csv"), DataFrame)
var_inf = CSV.read(joinpath(joinpath(@__DIR__, "..", "experiments", "variational_inference", "standard_nn", problem_name, experiment_name), "aggregated_predictions.csv"), DataFrame)

include(joinpath(joinpath(@__DIR__, "..", "experiments", "ensemble", "ensemble_10000", problem_name, "multistart_"*experiment_name), "experimental_setting.jl")) 
include("../experiments/$problem_name.jl")
include("../experiments/simulate.jl")
include("../experiments/utils.jl")
ref = solve(ODEProblem(simulation_dynamics!, u0, tspan, p_ode), saveat = saveat_plot)

custom_colors = palette(:tab10)[1:4] # get(ColorSchemes.viridis, [i/(n_colors-1) for i in 0:1:(n_colors-1)])
custom_bands = [RGB(165/255, 200/255, 225/255), RGB(255/255, 203/255, 158/255), RGB(170/255, 217/255, 170/255), RGB(238/255, 168/255, 169/255)]


fig = let 
    f = Figure(size = (1200, 400),  px_per_unit = 10)

    ga = f[1, 1] = GridLayout()

    # Define axis of first line
    ax11 = CairoMakie.Axis(ga[1, 2], title = "S", backgroundcolor=(:white), xlabelsize=20, 
        xticklabelsvisible=false) #, width=200)
    ax12 = CairoMakie.Axis(ga[1, 3], title = "E", backgroundcolor=(:white), xticklabelsvisible=false, yticklabelsvisible=false) #, width=200)      
    ax13 = CairoMakie.Axis(ga[1, 4], title = "I", backgroundcolor=(:white), xticklabelsvisible=false, yticklabelsvisible=false) #, width=200)      
    ax14 = CairoMakie.Axis(ga[1, 5], title = "R", backgroundcolor=(:white), xticklabelsvisible=false, yticklabelsvisible=false) #, width=200)

    ax1 = CairoMakie.Axis(ga[1,1], width = 100)
    hidedecorations!(ax1)  # hides ticks, grid and lables
    hidespines!(ax1)
    text!(ax1, "Ensemble", fontsize=15,space = :relative, font=:bold, position=Point2f(0.,0.6))
    
    ax3 = CairoMakie.Axis(ga[2,1], width = 100)
    hidedecorations!(ax3)  # hides ticks, grid and lables
    hidespines!(ax3)
    text!(ax3, "MCMC", fontsize=15,space = :relative, font=:bold, position=Point2f(0.,0.6))
    ax4 = CairoMakie.Axis(ga[3,1], width = 100)
    hidedecorations!(ax4)  # hides ticks, grid and lables
    hidespines!(ax4)
    text!(ax4, "Variational \nInference", fontsize=15,space = :relative, font=:bold, position=Point2f(0.,0.5))
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
    b_legend = band!(ax11, ensemble.t, ensemble.S_q_p5, ensemble.S_q_99p5, alpha=0.0, color=RGB(0.7,0.7,0.7))
    l_legend = lines!(ax11, ensemble.t, ensemble.S_mean, color=RGB(0.4, 0.4, 0.4), linewidth=3)
    s_legend = CairoMakie.scatter!(ax13, t_sim, X_sim[3,:], color=RGB(0.4, 0.4, 0.4), markersize=7)

    b11 = band!(ax11, ensemble.t, ensemble.S_q_p5, ensemble.S_q_99p5, alpha=0.0, color=custom_bands[1])
    l11 = lines!(ax11, ensemble.t, ensemble.S_mean, color=(custom_colors[1] + RGB(0,0,0)*0.2,1.0), linewidth=3)
    r11 = lines!(ax11, ref.t, Array(ref)[1,:], color=RGB(0,0,0), linewidth=2, linestyle=:dash)

    b12 = band!(ax12, ensemble.t, ensemble.E_q_p5, ensemble.E_q_99p5, alpha=0.01, color=custom_bands[2]) 
    l12 = lines!(ax12, ensemble.t, ensemble.E_mean, color=(custom_colors[2],1.0), linewidth=3)
    r12 = lines!(ax12, ref.t, Array(ref)[2,:], color=RGB(0,0,0), linewidth=2, linestyle=:dash)

    b13 = band!(ax13, ensemble.t, ensemble.I_q_p5, ensemble.I_q_99p5, alpha=0.01, color=custom_bands[3]) 
    l13 = lines!(ax13, ensemble.t, ensemble.I_mean, color=(custom_colors[3],1.0), linewidth=3)
    s13 = CairoMakie.scatter!(ax13, t_sim, X_sim[3,:], color = (custom_colors[3]-RGB(0.3,0.3,0.3),1), markersize=7)
    r13 = lines!(ax13, ref.t, Array(ref)[3,:], color=RGB(0,0,0), linewidth=2, linestyle=:dash)

    b14 = band!(ax14, ensemble.t, ensemble.R_q_p5, ensemble.R_q_99p5, alpha=0.01, color=custom_bands[4]) 
    l14 = lines!(ax14, ensemble.t, ensemble.R_mean, color=(custom_colors[4],1.0), linewidth=3)
    s14 = CairoMakie.scatter!(ax14, t_sim, X_sim[4,:], color = (custom_colors[4]-RGB(0.3,0.3,0.3),1), markersize=7)
    r14 = lines!(ax14, ref.t, Array(ref)[4,:], color=RGB(0,0,0), linewidth=2, linestyle=:dash)

    ####### line 3
    b31 = band!(ax31, mcmc.t, mcmc.S_q_p5, mcmc.S_q_99p5, alpha=0.0, color=(custom_colors[1],0.4))
    l31 = lines!(ax31, mcmc.t, mcmc.S_mean, color=(custom_colors[1] + RGB(0,0,0)*0.2,1.0), linewidth=3)
    r31 = lines!(ax31, ref.t, Array(ref)[1,:], color=RGB(0,0,0), linewidth=2, linestyle=:dash)

    b32 = band!(ax32, mcmc.t, mcmc.E_q_p5, mcmc.E_q_99p5, alpha=0.01, color=(custom_colors[2],0.4)) 
    l32 = lines!(ax32, mcmc.t, mcmc.E_mean, color=(custom_colors[2],1.0), linewidth=3)
    r32 = lines!(ax32, ref.t, Array(ref)[2,:], color=RGB(0,0,0), linewidth=2, linestyle=:dash)

    b33 = band!(ax33, mcmc.t, mcmc.I_q_p5, mcmc.I_q_99p5, alpha=0.01, color=(custom_colors[3],0.4)) 
    l33 = lines!(ax33, mcmc.t, mcmc.I_mean, color=(custom_colors[3],1.0), linewidth=3)
    s33 = CairoMakie.scatter!(ax33, t_sim, X_sim[3,:], color = (custom_colors[3]-RGB(0.3,0.3,0.3),1), markersize=7)
    r33 = lines!(ax33, ref.t, Array(ref)[3,:], color=RGB(0,0,0), linewidth=2, linestyle=:dash)

    b34 = band!(ax34, mcmc.t, mcmc.R_q_p5, mcmc.R_q_99p5, alpha=0.01, color=(custom_colors[4],0.4)) 
    l34 = lines!(ax34, mcmc.t, mcmc.R_mean, color=(custom_colors[4],1.0), linewidth=3)
    s34 = CairoMakie.scatter!(ax34, t_sim, X_sim[4,:], color = (custom_colors[4]-RGB(0.3,0.3,0.3),1), markersize=7)
    r34 = lines!(ax34, ref.t, Array(ref)[4,:], color=RGB(0,0,0), linewidth=2, linestyle=:dash)

    legend25 = Legend(ga[2, 6],
        [r11, s_legend, l_legend, b_legend], # [hist, vline],
        ["ground truth", "observed", "posterior mean", "99% prediction interval"], orientation = :vertical, framevisible=false)

    
    # legend51 = Legend(ga[4,2], [[l11, l31, l41], [b11,  b31]],
    #     ["ensemble mean", "ensemble area"], framevisible=false)
    # legend52 = Legend(ga[4,3], [[l12, l32, l42], [b12,  b32]],
    #     ["ensemble mean", "ensemble area"], framevisible=false)
    # legend53 = Legend(ga[4,4], [[l13, l33, l43], [b13,  b33], [s13, s33, s43]],
    #     ["ensemble mean", "ensemble area", "observed"], framevisible=false)
    # legend54 = Legend(ga[4,5], [[l14,  l34, l44], [b14, b34], [s14, s34, s44]],
    #     ["ensemble mean", "ensemble area", "observed"], framevisible=false)

    ############# line 4
    b41 = band!(ax41, var_inf.t, var_inf.S_q_p5, var_inf.S_q_99p5, alpha=0.0, color=(custom_colors[1],0.4))
    l41 = lines!(ax41, var_inf.t, var_inf.S_mean, color=(custom_colors[1] + RGB(0,0,0)*0.2,1.0), linewidth=3)
    r41 = lines!(ax41, ref.t, Array(ref)[1,:], color=RGB(0,0,0), linewidth=2, linestyle=:dash)

    b42 = band!(ax42, var_inf.t, var_inf.E_q_p5, var_inf.E_q_99p5, alpha=0.01, color=(custom_colors[2],0.4)) 
    l42 = lines!(ax42, var_inf.t, var_inf.E_mean, color=(custom_colors[2],1.0), linewidth=3)
    r42 = lines!(ax42, ref.t, Array(ref)[2,:], color=RGB(0,0,0), linewidth=2, linestyle=:dash)

    b43 = band!(ax43, var_inf.t, var_inf.I_q_p5, var_inf.I_q_99p5, alpha=0.01, color=(custom_colors[3],0.4)) 
    l43 = lines!(ax43, var_inf.t, var_inf.I_mean, color=(custom_colors[3],1.0), linewidth=3)
    s43 = CairoMakie.scatter!(ax43, t_sim, X_sim[3,:], color = (custom_colors[3]-RGB(0.3,0.3,0.3),1), markersize=7)
    r43 = lines!(ax43, ref.t, Array(ref)[3,:], color=RGB(0,0,0), linewidth=2, linestyle=:dash)

    b44 = band!(ax44, var_inf.t, var_inf.R_q_p5, var_inf.R_q_99p5, alpha=0.01, color=(custom_colors[4],0.4)) 
    l44 = lines!(ax44, var_inf.t, var_inf.R_mean, color=(custom_colors[4],1.0), linewidth=3)
    s44 = CairoMakie.scatter!(ax44, t_sim, X_sim[4,:], color = (custom_colors[4]-RGB(0.3,0.3,0.3),1), markersize=7)
    r44 = lines!(ax44, ref.t, Array(ref)[4,:], color=RGB(0,0,0), linewidth=2, linestyle=:dash)


    resize_to_layout!(f)

    f
end

save(joinpath("paper_figures", "plots", problem_name, experiment_name, "method_comparison.pdf"), fig)
