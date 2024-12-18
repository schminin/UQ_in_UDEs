using LinearAlgebra, Statistics
using DifferentialEquations # for automatic solver selection
using ComponentArrays, Lux, Zygote, Plots, StableRNGs, FileIO
using Distributions
using DataFrames, CSV
# gr()
using CairoMakie
using ColorSchemes

#========================================================================================================
    Define what experiment to run the analyis on
========================================================================================================#
# cluster settings
problem_name = ARGS[2] # "quadratic_dynamics"
experiment_name = ARGS[3] # "ground_truth_gaussian_noise_0.01"
experiment_series = ARGS[4]

# local settings
#experiment_series = "ensemble_10000"
#problem_name = "seir_pulse"
#experiment_name = "multistart_gaussian_IR_noise_0.01"

experiment_path = joinpath(@__DIR__, "..", experiment_series, problem_name, experiment_name)
include(joinpath(experiment_path, "experimental_setting.jl")) 
include(joinpath(@__DIR__, "plot_utils.jl")) 
include(joinpath(@__DIR__, "../../evaluation_utils.jl")) 

#========================================================================================================
    Ensemble Evaluation
========================================================================================================#
df_trajectory = CSV.read(joinpath(experiment_path, "ensemble_trajectory.csv"), DataFrame)
df_stats = CSV.read(joinpath(experiment_path, "ensemble_stats.csv"), DataFrame)

# report number of failed model optimizations
report_fails(df_stats, n_models)

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
