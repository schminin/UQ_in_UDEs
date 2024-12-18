

# Import libraries
using Lux, Turing, CairoMakie, Random, StableRNGs
using Turing: Variational
using OrdinaryDiffEq, DifferentialEquations, ModelingToolkit, SciMLSensitivity
using ComponentArrays, Functors
using JLD2, LinearAlgebra

#========================================================================================================
    Experimental Settings
========================================================================================================#
experiment_series = "standard_nn/"
problem_name = ARGS[2] # "seir"
experiment_name = ARGS[3] #  "negbin_IR_1.2"

# create paths and load experimental setting
experiment_path = joinpath(pwd(), "experiments/variational_inference", experiment_series, problem_name, experiment_name)
include(joinpath(experiment_path, "experimental_setting.jl"))
isdir(joinpath(experiment_path, "log")) || mkpath(joinpath(experiment_path, "log"))
isdir(joinpath(experiment_path, "posteriors")) || mkpath(joinpath(experiment_path, "posteriors"))

#========================================================================================================
    Data creation
========================================================================================================#
include("../$problem_name.jl")
include("../simulate.jl")
# visualise_simulation(sample_data; model_id = 1)

#========================================================================================================
    UDE utils
========================================================================================================#
include("../utils.jl")

#========================================================================================================
    VI utils
========================================================================================================#
# The Bayesian model definition is same as for the MCMC experiments
if occursin("seir", problem_name)
    if noise_model=="negBin"
        include("../MCMC/mcmc_seir_negbin.jl")
    else
        include("../MCMC/mcmc_seir.jl")
    end
else
    include("../MCMC/mcmc_$problem_name.jl")
end

include("vi_utils.jl")

# Perform inference.
nn_model, p_init, st = prepare_model(hyperpar, 1, sample_mechanistic, noise_model, true, component_vector=false)
ude_prob = ODEProblem(dynamics!, u0, tspan, p_init)
t_obs, X_obs, t_weights = sample_data(1)

rng = StableRNG(42);
model = bayes_nn(X_obs[[1],:], ude_prob)

# results with the mean field assumption
advi_3 = ADVI(30, 10000)
q_3 = vi(model, advi_3);
save(joinpath(experiment_path, "posteriors", "advi_10000.jld2"), "q", q_3)
print("3 finished")


advi_4 = ADVI(30, 100000)
q_4 = vi(model, advi_4);
save(joinpath(experiment_path, "posteriors", "advi_100000.jld2"), "q", q_4)
print("4 finished")


q = load(joinpath(experiment_path, "posteriors", "advi_100000.jld2"))["q"]
sampled_data = rand(q, 10000)




using CairoMakie
using Plots
using ColorSchemes

isdir(joinpath(experiment_path, "plots")) || mkpath(joinpath(experiment_path, "plots"))

function density_plot(parameter_name, x_vals, vertical_line_at; x_color=:steelblue4, store_fig=true)
    density_fig = let
        f = Figure(fonts = (; regular = "Dejavu"))
        ax = CairoMakie.Axis(f[1, 1], title = "Distribution of $parameter_name", xlabel = "$parameter_name")
        hist = CairoMakie.density!(ax, x_vals, normalization = :pdf, 
            label_size = 15, strokecolor = (:white, 0.5), color=x_color) 
        vline = vlines!(ax, [vertical_line_at]; color=:tomato4)
        Legend(f[1, 2],
            [hist, vline],
            ["distribution", "reference"])
        CairoMakie.ylims!(low=0)
        f
    end
    density_fig
    if store_fig
        save(joinpath(experiment_path, "plots", "density_$parameter_name.png"), density_fig)
    else
        return density_fig
    end
end

density_with_reference(p_ode[1], "α", retrieve_α.(sampled_data[1,:]), RGB(0.21, 0.39, 0.55); plot_path=save(joinpath(experiment_path, "plots")))

density_plot("alpha", retrieve_α.(sampled_data[1,:]), p_ode[1]; x_color= RGB(0.21, 0.39, 0.55))
density_plot("gamma", retrieve_γ.(sampled_data[2,:]), p_ode[2]; x_color= RGB(0.32, 0.55, 0.55))
density_plot("sigma", exp.(sampled_data[3,:]), 0.01; x_color=RGB(0.68, 0.85, 0.9))
fig = let
    f = Figure(size = (1000, 300))
    ga = f[1, 1] = GridLayout()

    # Distribution of alpha
    ax1 = CairoMakie.Axis(ga[1, 1], title = "α", backgroundcolor=(:white), xlabelsize=20)
    dens1 = CairoMakie.density!(ax1, retrieve_α.(sampled_data[1,:]), normalization = :pdf, 
            label_size = 15, strokecolor = (:white, 0.5), color=(RGB(0.21, 0.39, 0.55), 0.7)) 
    vline = vlines!(ax1, [p_ode[1]]; color=:tomato4, linewidth=2)
    CairoMakie.ylims!(low=0)
    ax2 = CairoMakie.Axis(ga[1, 2], title = "γ", backgroundcolor=(:white), xlabelsize=20)
    dens2 = CairoMakie.density!(ax2, retrieve_γ.(sampled_data[2,:]), normalization = :pdf, 
            label_size = 15, strokecolor = (:white, 0.5), color=(RGB(0.32, 0.55, 0.55), 0.7)) 
    vline = vlines!(ax2, [p_ode[2]]; color=:tomato4, linewidth=2)
    CairoMakie.ylims!(low=0)
    if noise_model=="Gaussian"
        global par_name = "σ"
    elseif noise_model=="negBin"
        global par_name = "overdispersion parameter"
    end
    ax3 = CairoMakie.Axis(ga[1, 3], title = par_name, backgroundcolor=(:white), xlabelsize=20)
    dens3 = CairoMakie.density!(ax3, sqrt.(exp.(sampled_data[3,:])), normalization = :pdf, 
            label_size = 15, strokecolor = (:white, 0.5), color=(RGB(0.68, 0.85, 0.9), 0.7)) 
    vline = vlines!(ax3, [noise_par]; color=:tomato4, linewidth=2)
    CairoMakie.ylims!(low=0)

    Legend(ga[1, 4],
        [[dens1, dens2, dens3], vline],
        ["distribution", "reference"])
    f
end
save(joinpath(experiment_path, "plots", "const_parameters_overview.png"), fig)


df_stats = DataFrame()
df_trajectory = DataFrame()
for (sample_id, p) in enumerate(eachcol(Array(sampled_data)))
    par = merge((par_α = p[1], par_γ=p[2], np=sqrt.(exp.(p[3]))), (nn=vector_to_parameters(p[4:end], p_init.nn),))
    df_stats_sub, df_trajectory_sub = evaluate_parameter_sample(sample_id, par)
    append!(df_stats, df_stats_sub, cols=:union)
    append!(df_trajectory, df_trajectory_sub, cols=:union)
end
CSV.write(joinpath(experiment_path, "posterior_stats.csv"), df_stats)
CSV.write(joinpath(experiment_path, "posterior_trajectory.csv"), df_trajectory)

include("../evaluation_utils.jl")
pred_agg = aggregate_predictions(CSV.read(joinpath(experiment_path, "posterior_trajectory.csv"), DataFrame), 1, problem_name)
CSV.write(joinpath(experiment_path, "aggregated_predictions.csv"), pred_agg)



pred = []
β_pred = []
for i in 1:size(sampled_data)[2]
    p = sampled_data[:,i]
    nn_par = vector_to_parameters(p[4:end], p_init.nn)
    par = merge((par_α = p[1], par_γ=p[2]), (nn=nn_par,))
    ude_prob = ODEProblem(dynamics!, u0, tspan, par)
    sol = solve(ude_prob, saveat=0:0.5:126)
    sol_β = retrieve_β.(nn_model(Array(0:0.5:126)', nn_par, st)[1])
    push!(β_pred, sol_β[1,:])
    push!(pred, Array(sol))
end
pred_array = reduce((x,y) -> cat(x, y, dims=3), pred)
pred_array_β = reduce((x,y) -> hcat(x, y), pred)


custom_colors = palette(:tab10)[1:4] # get(ColorSchemes.viridis, [i/(n_colors-1) for i in 0:1:(n_colors-1)])
trajectory_plot = let
    g = Figure(fonts = (; regular = "Dejavu"))
    ax = CairoMakie.Axis(g[1, 1], title = "Trajectories", xlabel = "time", backgroundcolor=(:white))
    b1 = band!(ax, 0:0.5:126, minimum(pred_array[1,:,:], dims=2)[:,1], maximum(pred_array[1,:,:], dims=2)[:,1], alpha=0.0, color=(custom_colors[1],0.5))
    b2 = band!(ax, 0:0.5:126, minimum(pred_array[2,:,:], dims=2)[:,1], maximum(pred_array[2,:,:], dims=2)[:,1], alpha=0.01, color=(custom_colors[2],0.5)) 
    b3 = band!(ax, 0:0.5:126, minimum(pred_array[3,:,:], dims=2)[:,1], maximum(pred_array[3,:,:], dims=2)[:,1], alpha=0.01, color=(custom_colors[3],0.5)) 
    b4 = band!(ax, 0:0.5:126, minimum(pred_array[4,:,:], dims=2)[:,1], maximum(pred_array[4,:,:], dims=2)[:,1], alpha=0.01, color=(custom_colors[4],0.5)) 
    #s1 = CairoMakie.scatter!(ax, t_sim, X_sim[1,:], color = (custom_colors[1],1), markersize=6)
    #s2 = CairoMakie.scatter!(ax, t_sim, X_sim[2,:], color = (custom_colors[2],1), markersize=6)
    s3 = CairoMakie.scatter!(ax, t_sim, X_sim[3,:], color = (custom_colors[3]-RGB(0.3,0.3,0.3),1), markersize=6)
    s4 = CairoMakie.scatter!(ax, t_sim, X_sim[4,:], color = (custom_colors[4]-RGB(0.3,0.3,0.3),1), markersize=6)
    l1 = lines!(ax, 0:0.5:126, mean(pred_array[1,:,:], dims=2)[:,1], color=(custom_colors[1],1.0))
    l2 = lines!(ax, 0:0.5:126, mean(pred_array[2,:,:], dims=2)[:,1], color=(custom_colors[2],1.0))
    l3 = lines!(ax, 0:0.5:126, mean(pred_array[3,:,:], dims=2)[:,1], color=(custom_colors[3],1.0))
    l4 = lines!(ax, 0:0.5:126, mean(pred_array[4,:,:], dims=2)[:,1], color=(custom_colors[4],1.0))
    legend = Legend(g[1,2],
    [[b1, l1], [b2, l2], [b3, l3], [b4,l4], [s3, s4]], # [hist, vline],
    ["S", "E", "I", "R", "observed"])
    g
end
save(joinpath(experiment_path, "plots","trajectory_plot.png"), trajectory_plot)

ref = solve(ODEProblem(seir!, u0, tspan, p_ode), saveat = 0.5)
trajectory_plot = let
    g = Figure(fonts = (; regular = "Dejavu"))
    ax = CairoMakie.Axis(g[1, 1], title = "Trajectories", xlabel = "time", backgroundcolor=(:white))
    b1 = band!(ax, 0:0.5:126, minimum(pred_array[1,:,:], dims=2)[:,1], maximum(pred_array[1,:,:], dims=2)[:,1], alpha=0.0, color=(custom_colors[1],0.5))
    b2 = band!(ax, 0:0.5:126, minimum(pred_array[2,:,:], dims=2)[:,1], maximum(pred_array[2,:,:], dims=2)[:,1], alpha=0.01, color=(custom_colors[2],0.5)) 
    b3 = band!(ax, 0:0.5:126, minimum(pred_array[3,:,:], dims=2)[:,1], maximum(pred_array[3,:,:], dims=2)[:,1], alpha=0.01, color=(custom_colors[3],0.5)) 
    b4 = band!(ax, 0:0.5:126, minimum(pred_array[4,:,:], dims=2)[:,1], maximum(pred_array[4,:,:], dims=2)[:,1], alpha=0.01, color=(custom_colors[4],0.5)) 
    s1 = lines!(ax, ref.t, Array(ref)[1,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)
    s2 = lines!(ax, ref.t, Array(ref)[2,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)
    s3 = lines!(ax, ref.t, Array(ref)[3,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)
    s4 = lines!(ax, ref.t, Array(ref)[4,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)
    l1 = lines!(ax, 0:0.5:126, median(pred_array[1,:,:], dims=2)[:,1], color=(custom_colors[1],1.0))
    l2 = lines!(ax, 0:0.5:126, median(pred_array[2,:,:], dims=2)[:,1], color=(custom_colors[2],1.0))
    l3 = lines!(ax, 0:0.5:126, median(pred_array[3,:,:], dims=2)[:,1], color=(custom_colors[3],1.0))
    l4 = lines!(ax, 0:0.5:126, median(pred_array[4,:,:], dims=2)[:,1], color=(custom_colors[4],1.0))
    legend = Legend(g[1,2],
    [[b1, l1], [b2, l2], [b3, l3], [b4,l4], [s1, s2, s3, s4]], # [hist, vline],
    ["S", "E", "I", "R", "reference"])
    g
end
save(joinpath(experiment_path, "plots", "trajectory_reference.png"), trajectory_plot)




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
    b11 = band!(ax11, 0:0.5:126, minimum(pred_array[1,:,:], dims=2)[:,1], maximum(pred_array[1,:,:], dims=2)[:,1], alpha=0.0, color=(custom_colors[1],0.5))
    l11 = lines!(ax11, 0:0.5:126, mean(pred_array[1,:,:], dims=2)[:,1], color=(custom_colors[1],1.0), linewidth=3)
    r11 = lines!(ax11, ref.t, Array(ref)[1,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

    b12 = band!(ax12, 0:0.5:126, minimum(pred_array[2,:,:], dims=2)[:,1], maximum(pred_array[2,:,:], dims=2)[:,1], alpha=0.01, color=(custom_colors[2],0.5)) 
    l12 = lines!(ax12, 0:0.5:126, mean(pred_array[2,:,:], dims=2)[:,1], color=(custom_colors[2],1.0), linewidth=3)
    r12 = lines!(ax12, ref.t, Array(ref)[2,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

    b13 = band!(ax13, 0:0.5:126, minimum(pred_array[3,:,:], dims=2)[:,1], maximum(pred_array[3,:,:], dims=2)[:,1], alpha=0.01, color=(custom_colors[3],0.5)) 
    l13 = lines!(ax13, 0:0.5:126, mean(pred_array[3,:,:], dims=2)[:,1], color=(custom_colors[3],1.0), linewidth=3)
    s13 = CairoMakie.scatter!(ax13, t_sim, X_sim[3,:], color = (custom_colors[3]-RGB(0.3,0.3,0.3),1), markersize=5)
    r13 = lines!(ax13, ref.t, Array(ref)[3,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

    b14 = band!(ax14, 0:0.5:126, minimum(pred_array[4,:,:], dims=2)[:,1], maximum(pred_array[4,:,:], dims=2)[:,1], alpha=0.01, color=(custom_colors[4],0.5)) 
    l14 = lines!(ax14, 0:0.5:126, mean(pred_array[4,:,:], dims=2)[:,1], color=(custom_colors[4],1.0), linewidth=3)
    s14 = CairoMakie.scatter!(ax14, t_sim, X_sim[4,:], color = (custom_colors[4]-RGB(0.3,0.3,0.3),1), markersize=5)
    r14 = lines!(ax14, ref.t, Array(ref)[4,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

    legend15 = Legend(ga[1, 6],
        [[r11, r12, r13, r14]], # [hist, vline],
        ["reference"], orientation = :vertical, framevisible=false)

    # Define content of second line
    b21 = band!(ax21, 0:0.5:126, minimum(pred_array[1,:,:], dims=2)[:,1], maximum(pred_array[1,:,:], dims=2)[:,1], alpha=0.0, color=(custom_colors[1],0.5))
    #l21 = lines!(ax21, traj_comb.t, traj_comb.S_opt_mean, color=(custom_colors[1],1.0), linewidth=3)
    #gtb21 = band!(ax21, traj_comb_ground_truth.t, traj_comb_ground_truth.S_opt_minimum, traj_comb_ground_truth.S_opt_maximum, alpha=0.0, color=(:grey,0.5))
    #gtl21 = lines!(ax21, traj_comb_ground_truth.t, traj_comb_ground_truth.S_opt_mean, color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

    b22 = band!(ax22, 0:0.5:126, minimum(pred_array[2,:,:], dims=2)[:,1], maximum(pred_array[2,:,:], dims=2)[:,1], alpha=0.01, color=(custom_colors[2],0.5)) 
    #l22 = lines!(ax22, traj_comb.t, traj_comb.E_opt_mean, color=(custom_colors[2],1.0), linewidth=3)
    #gtb22 = band!(ax22, traj_comb_ground_truth.t, traj_comb_ground_truth.E_opt_minimum, traj_comb_ground_truth.E_opt_maximum, alpha=0.0, color=(:grey,0.5))
    #gtl22 = lines!(ax22, traj_comb_ground_truth.t, traj_comb_ground_truth.E_opt_mean, color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

    b23 = band!(ax23, 0:0.5:126, minimum(pred_array[3,:,:], dims=2)[:,1], maximum(pred_array[3,:,:], dims=2)[:,1], alpha=0.01, color=(custom_colors[3],0.5)) 
    #l23 = lines!(ax23, traj_comb.t, traj_comb.I_opt_mean, color=(custom_colors[3],1.0), linewidth=3)
    #s23 = CairoMakie.scatter!(ax23, t_sim, X_sim[3,:], color = (custom_colors[3]-RGB(0.3,0.3,0.3),1), markersize=5)
    #gtb23 = band!(ax23, traj_comb_ground_truth.t, traj_comb_ground_truth.I_opt_minimum, traj_comb_ground_truth.I_opt_maximum, alpha=0.0, color=(:grey,0.5))
    #gtl23 = lines!(ax23, traj_comb_ground_truth.t, traj_comb_ground_truth.I_opt_mean, color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

    b24 = band!(ax24, 0:0.5:126, minimum(pred_array[4,:,:], dims=2)[:,1], maximum(pred_array[4,:,:], dims=2)[:,1], alpha=0.01, color=(custom_colors[4],0.5)) 
    #s24 = CairoMakie.scatter!(ax24, t_sim, X_sim[4,:], color = (custom_colors[4]-RGB(0.3,0.3,0.3),1), markersize=5)
    #l24 = lines!(ax24, traj_comb.t, traj_comb.R_opt_mean, color=(custom_colors[4],1.0), linewidth=3)
    #gtb24 = band!(ax24, traj_comb_ground_truth.t, traj_comb_ground_truth.R_opt_minimum, traj_comb_ground_truth.R_opt_maximum, alpha=0.0, color=(:grey,0.5))
    #gtl24 = lines!(ax24, traj_comb_ground_truth.t, traj_comb_ground_truth.R_opt_mean, color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

    #legend25 = Legend(ga[2, 6],
    #    [[gtl21, gtl22, gtl23, gtl24], [gtb21, gtb22, gtb23, gtb24]], # [hist, vline],
    #    ["ground truth mean", "ground truth area"], orientation = :vertical, framevisible=false)

    ####### line 3


    ############# line 4


    legend51 = Legend(ga[5,2], [[l11], [b11]],
        ["ensemble mean", "ensemble area"], framevisible=false)
    legend52 = Legend(ga[5,3], [[l12], [b12]],
        ["ensemble mean", "ensemble area"], framevisible=false)
    legend53 = Legend(ga[5,4], [[l13], [b13], [s13]],
        ["ensemble mean", "ensemble area", "observed"], framevisible=false)
    legend54 = Legend(ga[5,5], [[l14], [b14], [s14]],
        ["ensemble mean", "ensemble area", "observed"], framevisible=false)

    resize_to_layout!(f)

    f
end

save(joinpath(experiment_path, "plots", "states_overview.png"), trajectory_summary)
