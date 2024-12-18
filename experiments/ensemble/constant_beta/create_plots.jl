using LinearAlgebra, Statistics
using DifferentialEquations # for automatic solver selection
using ComponentArrays, Lux, Zygote, Plots, StableRNGs, FileIO
using Distributions
using DataFrames, CSV
# gr()
using CairoMakie
using ColorSchemes

#========================================================================================================
    Ensemble Evaluation
========================================================================================================#
experiment_series = "constant_beta"
problem_name = "seir"
experiment_name = "multistart_gaussian_IR_noise_0.01"

# create paths and load experimental setting
experiment_path = joinpath(pwd(), "experiments/ensemble", experiment_series, problem_name, experiment_name)

df_trajectory = CSV.read(joinpath(experiment_path, "ensemble_trajectory.csv"), DataFrame)
df_stats = CSV.read(joinpath(experiment_path, "ensemble_stats.csv"), DataFrame)
pred_agg = CSV.read(joinpath(experiment_path, "aggregated_predictions.csv"), DataFrame)

# minimum negLL
minimum(skipmissing(df_stats.negLL_val_p_opt))
minimum(skipmissing(df_stats.negLL_train_p_opt))
minimum(skipmissing(df_stats.negLL_obs_p_opt))

# find cutoff value based of likelihood ratio test
# 95% quantile of xi² distribution with 1 degree of freedom
cutoff_chi_95 = quantile(Distributions.Chisq(1),0.95)/2 + minimum(skipmissing(df_stats.negLL_obs_p_opt))
println("cutoff_chi_95 = $cutoff_chi_95")
println("models = $(sum(df_stats.negLL_obs_p_opt .< cutoff_chi_95))")


# find cutoff value based on 95% of best-performing models
cutoff_95 = quantile(skipmissing(df_stats.negLL_obs_p_opt), 0.95)


#=======================================================================================================
    plot predictions next to each other
========================================================================================================#
include(joinpath(experiment_path, "experimental_setting.jl"))
include("$problem_name.jl")
include("../../simulate.jl")

t_sim, X_sim = simulate_observations(n_timepoints, noise_model, noise_par, 1, 1; ode_prob = simulation_prob)  
ref = solve(ODEProblem(simulation_dynamics!, u0, tspan, p_ode), saveat = saveat_plot)

custom_colors = palette(:tab10)[1:4] # get(ColorSchemes.viridis, [i/(n_colors-1) for i in 0:1:(n_colors-1)])
custom_bands = [RGB(165/255, 200/255, 225/255), RGB(255/255, 203/255, 158/255), RGB(170/255, 217/255, 170/255), RGB(238/255, 168/255, 169/255)]


fig = let 
    f = Figure(size = (1200, 200),  px_per_unit = 10)

    ga = f[1, 1] = GridLayout()

    # Define axis of first line
    ax11 = CairoMakie.Axis(ga[1, 1], title = "S", backgroundcolor=(:white), xlabel = "time") #, width=200)
    ax12 = CairoMakie.Axis(ga[1, 2], title = "E", backgroundcolor=(:white), xlabel = "time", yticklabelsvisible=false) #, width=200)      
    ax13 = CairoMakie.Axis(ga[1, 3], title = "I", backgroundcolor=(:white), xlabel = "time", yticklabelsvisible=false) #, width=200)      
    ax14 = CairoMakie.Axis(ga[1, 4], title = "R", backgroundcolor=(:white), xlabel = "time", yticklabelsvisible=false) #, width=200)
   
    f

    linkyaxes!(ax11, ax12, ax13, ax14)


    # Define content of first line
    b_legend = band!(ax11, pred_agg.t, pred_agg.S_q_p5, pred_agg.S_q_99p5, alpha=0.0, color=RGB(0.7,0.7,0.7))
    l_legend = lines!(ax11, pred_agg.t, pred_agg.S_mean, color=RGB(0.4, 0.4, 0.4), linewidth=3)
    s_legend = CairoMakie.scatter!(ax13, t_sim, X_sim[3,:], color=RGB(0.4, 0.4, 0.4), markersize=7)

    b11 = band!(ax11, pred_agg.t, pred_agg.S_q_p5, pred_agg.S_q_99p5, alpha=0.0, color=custom_bands[1])
    l11 = lines!(ax11, pred_agg.t, pred_agg.S_mean, color=(custom_colors[1] + RGB(0,0,0)*0.2,1.0), linewidth=3)
    r11 = lines!(ax11, ref.t, Array(ref)[1,:], color=RGB(0,0,0), linewidth=2, linestyle=:dash)

    b12 = band!(ax12, pred_agg.t, pred_agg.E_q_p5, pred_agg.E_q_99p5, alpha=0.01, color=custom_bands[2]) 
    l12 = lines!(ax12, pred_agg.t, pred_agg.E_mean, color=(custom_colors[2],1.0), linewidth=3)
    r12 = lines!(ax12, ref.t, Array(ref)[2,:], color=RGB(0,0,0), linewidth=2, linestyle=:dash)

    b13 = band!(ax13, pred_agg.t, pred_agg.I_q_p5, pred_agg.I_q_99p5, alpha=0.01, color=custom_bands[3]) 
    l13 = lines!(ax13, pred_agg.t, pred_agg.I_mean, color=(custom_colors[3],1.0), linewidth=3)
    s13 = CairoMakie.scatter!(ax13, t_sim, X_sim[3,:], color = (custom_colors[3]-RGB(0.3,0.3,0.3),1), markersize=7)
    r13 = lines!(ax13, ref.t, Array(ref)[3,:], color=RGB(0,0,0), linewidth=2, linestyle=:dash)

    b14 = band!(ax14, pred_agg.t, pred_agg.R_q_p5, pred_agg.R_q_99p5, alpha=0.01, color=custom_bands[4]) 
    l14 = lines!(ax14, pred_agg.t, pred_agg.R_mean, color=(custom_colors[4],1.0), linewidth=3)
    s14 = CairoMakie.scatter!(ax14, t_sim, X_sim[4,:], color = (custom_colors[4]-RGB(0.3,0.3,0.3),1), markersize=7)
    r14 = lines!(ax14, ref.t, Array(ref)[4,:], color=RGB(0,0,0), linewidth=2, linestyle=:dash)

    legend25 = Legend(ga[1, 5],
        [r11, s_legend, l_legend, b_legend], # [hist, vline],
        ["ground truth", "observed", "posterior mean", "99% prediction interval"], orientation = :vertical, framevisible=false)

    f
end
save(joinpath("paper_figures", "plots", "constant_beta.pdf"), fig)


#=======================================================================================================
    calculate R²
========================================================================================================#

if noise_model=="Gaussian"
    X_true = Array(solve(simulation_prob, Tsit5(), abstol = 1e-12, reltol = 1e-12, saveat = sort(unique(df_trajectory.t))))

    if problem_name in ["seir_pulse", "seir"]
        col_names = ["S_opt", "E_opt", "I_opt", "R_opt"]
    else
        col_names = ["x_opt"]
    end

    for col in col_names
        df_stats[!, "R2_"*col] = [3.0 for i in 1:nrow(df_stats)]
    end

    for model_id in 1:n_models
        df_sub = df_trajectory[df_trajectory.model_id .== model_id, :]
        if nrow(df_sub)>1
            for (i, col) in enumerate(col_names)
                R² = R_square(df_sub[!, col], X_true[i,:])
                df_stats[df_stats.model_id.==model_id, "R2_"*col] = [R²]
            end
        end
    end
end

#========================================================================================================
    Visual Inspection
========================================================================================================#

isdir(joinpath(experiment_path, "plots")) || mkpath(joinpath(experiment_path, "plots"))

evaluation_list = [(cutoff_95, "minimal_95"), (cutoff_chi_95, "chi_2_95"), (cutoff_chi_95, "chi_2_95_alpha_1")]


# create examplary simulated data if not done already
t_sim, X_sim = simulate_observations(n_timepoints, noise_model, noise_par, 1, 1; ode_prob = simulation_prob)  

for (cutoff, cutoff_name) in evaluation_list
    plot_path = joinpath(experiment_path, "plots", cutoff_name)
    isdir(plot_path) || mkpath(plot_path)

    df_sub = dropmissing(df_stats, :negLL_obs_p_opt)

    df_sub = df_sub[df_sub.negLL_obs_p_opt .<= cutoff, :]
    if cutoff_name=="chi_2_95_alpha_1"
        df_sub = df_sub[df_sub.α_opt .<= 1., :]
        cutoff = maximum(df_sub.negLL_obs_p_opt)
    end
    println("$cutoff_name: $(nrow(df_sub)) observations")
    
    if problem_name in ["seir", "seir_pulse"]
        # Distribution plots
        seir_distribution_plots(plot_path, df_sub)
        # Prediction plots
        seir_trajectory_plots(plot_path, df_sub)
        alpha_beta_trajectory(plot_path, df_trajectory, df_sub)
    elseif problem_name=="quadratic_dynamics"
        quadratic_dynamics_density_plots(plot_path, df_sub, df_stats)
        quadratic_dynamics_trajectory_plots(plot_path, df_sub)
    end
    # Waterfall plot
    waterfall_plot = let
        df_sub = dropmissing(df_stats, :negLL_obs_p_opt)
        ys = sort(df_sub.negLL_obs_p_opt) .- minimum(df_sub.negLL_obs_p_opt)
        xs = 1:length(ys)
        f = Figure(fonts = (; regular = "Dejavu"))
        negLL_min = floor(minimum(ys), digits=0)
        negLL_max = ceil(maximum(ys), digits=0)
        ax = CairoMakie.Axis(f[1, 1], title = "Waterfall Plot", ylabel="Difference to Minimum Negative Log Likelihood",
            yticks = [negLL_min, round((negLL_max+negLL_min)/2, digits=0), negLL_max])
        scatters = scatterlines!(ax, xs, ys, color = :black, markercolor = ys)
        cutoff_line = hlines!(ax, [cutoff .- minimum(df_sub.negLL_obs_p_opt)], color=:tomato)
        desc = text!(ax, "cutoff value: $(round(cutoff.- minimum(df_sub.negLL_obs_p_opt), digits=1))", position=(0, cutoff.- minimum(df_sub.negLL_obs_p_opt) + 0.5), color=:tomato)
        f
    end
    save(joinpath(plot_path,"waterfall.png"), waterfall_plot)

    waterfall_plot = let
        df_sub = dropmissing(df_stats, :negLL_obs_p_opt)
        ys = sort(df_sub.negLL_obs_p_opt) .- minimum(df_sub.negLL_obs_p_opt)
        ys = sort(ys)[1:300]
        xs = 1:length(ys)
        f = Figure(fonts = (; regular = "Dejavu"))
        #negLL_min = floor(minimum(ys), digits=2)
        #negLL_max = ceil(maximum(ys), digits=2)
        ax = CairoMakie.Axis(f[1, 1], title = "Waterfall Plot (best 300)", ylabel="Difference to Minimum Negative Log Likelihood")
        scatters = scatterlines!(xs, ys, color = :black, markercolor = ys)
        f
    end
    save(joinpath(plot_path,"waterfall_zoom_300.png"), waterfall_plot)

    if noise_model == "Gaussian"
        r2_plot(col_names, df_stats, plot_path)
    end
end
