#========================================================================================================
    general: summary files
========================================================================================================#

struct ModelData
    pred_train::Union{Missing, Matrix{Float64}}
    X_train::Union{Missing, Matrix{Float64}}
    pred_val::Union{Missing, Matrix{Float64}}
    X_val::Union{Missing, Matrix{Float64}}
    pred_obs::Matrix{Float64}
    X_obs::Matrix{Float64}
    t_plot::Vector{Float64}
    pred_plot::Matrix{Float64}
end

function calculate_model_trajectories(data_id, model_id, p; 
        train_val_split,
        hyperpar=hyperpar, n_val=n_val, sample_mechanistic=sample_mechanistic, 
        noise_model=noise_model, random_init=random_init)
    # model trajectories
    t_obs, X_obs, t_weights = sample_data(data_id)
    nn_model, p_init, st = prepare_model(hyperpar, model_id, sample_mechanistic, noise_model, random_init)
    IC = u0

    function predict(p, saveat = t_train)  
        _prob = remake(prob_ude, p = p)
        Array(solve(_prob, Tsit5(), abstol = 1e-8, reltol = 1e-8, saveat = saveat))
    end

    prob_ude = ODEProblem(dynamics!, IC, tspan, p_init)

    pred_train = missing
    pred_val = missing
    X_train = missing
    X_val = missing
    if train_val_split
        t_train, X_train, t_val, X_val = train_validation_split(t_obs, X_obs, n_val, model_id)
        pred_train = predict(p, t_train)
        pred_val = predict(p, t_val)
    end
   
    pred_obs = predict(p, t_obs)
    pred_plot = predict(p, tspan[1]:saveat_plot:tspan[end])
    return ModelData(pred_train, X_train, pred_val, X_val, pred_obs, X_obs, tspan[1]:saveat_plot:tspan[end], pred_plot)
end

function aggregate_predictions(model_stats, trajectory_pred, data_id, problem_name, chi2_significance_level=0.05)
    model_stats_sub = model_stats[model_stats.data_id .== data_id, :]
    # get cutoff value
    cutoff_chi_2 = get_cutoff(model_stats_sub, chi2_significance_level)
    # append negLL information to prediction data
    df_merged = leftjoin(trajectory_pred, model_stats_sub[!,["data_id", "model_id", "negLL_obs_p_opt"]], on=["data_id","model_id"])
    # select the relevant trajectories
    pred_sub = df_merged[df_merged.data_id .== data_id, :]
    pred_sub = pred_sub[pred_sub.negLL_obs_p_opt .<= cutoff_chi_2, :]
    # aggregate the prediction information
    pred_agg = DataFrame()
    if occursin("seir", problem_name)
        pred_agg = seir_aggregation(pred_sub)
    else
        pred_agg = quadratic_dynamics_aggregation(pred_sub)
    end
    return pred_agg
end


function seir_aggregation(pred_sub)
    pred_agg = combine(groupby(pred_sub, "t"),
            :S => minimum, :S => mean, :S => maximum,
            :S => (q -> quantile(q, 0.005)) => :S_q_p5, # for 99% prediction intervals
            :S => (q -> quantile(q, 0.025)) => :S_q_2p5, # for 95% prediction intervals
            :S => (q -> quantile(q, 0.05)) => :S_q_5, # for 90% prediction intervals
            :S => (q -> quantile(q, 0.1)) => :S_q_10,
            :S => (q -> quantile(q, 0.15)) => :S_q_15, # for 70 % prediction intervals
            :S => (q -> quantile(q, 0.25)) => :S_q_25, 
            :S => (q -> quantile(q, 0.75)) => :S_q_75,
            :S => (q -> quantile(q, 0.85)) => :S_q_85,    
            :S => (q -> quantile(q, 0.9)) => :S_q_90,
            :S => (q -> quantile(q, 0.95)) => :S_q_95, # for 90% prediction intervals
            :S => (q -> quantile(q, 0.975)) => :S_q_97p5, # for 95% prediction intervals
            :S => (q -> quantile(q, 0.995)) => :S_q_99p5, # for 99% prediction intervals

            :E => minimum, :E => mean, :E => maximum,
            :E => (q -> quantile(q, 0.005)) => :E_q_p5, # for 99% prediction intervals
            :E => (q -> quantile(q, 0.025)) => :E_q_2p5, # for 95% prediction intervals
            :E => (q -> quantile(q, 0.05)) => :E_q_5, # for 90% prediction intervals
            :E => (q -> quantile(q, 0.1)) => :E_q_10,
            :E => (q -> quantile(q, 0.15)) => :E_q_15, # for 70 % prediction intervals
            :E => (q -> quantile(q, 0.25)) => :E_q_25, 
            :E => (q -> quantile(q, 0.75)) => :E_q_75,
            :E => (q -> quantile(q, 0.85)) => :E_q_85,    
            :E => (q -> quantile(q, 0.9)) => :E_q_90,
            :E => (q -> quantile(q, 0.95)) => :E_q_95, # for 90% prediction intervals
            :E => (q -> quantile(q, 0.975)) => :E_q_97p5, # for 95% prediction intervals
            :E => (q -> quantile(q, 0.995)) => :E_q_99p5, # for 99% prediction intervals
            :I => minimum, :I => mean, :I => maximum,
            :I => (q -> quantile(q, 0.005)) => :I_q_p5, # for 99% prediction intervals
            :I => (q -> quantile(q, 0.025)) => :I_q_2p5, # for 95% prediction intervals
            :I => (q -> quantile(q, 0.05)) => :I_q_5, # for 90% prediction intervals
            :I => (q -> quantile(q, 0.1)) => :I_q_10,
            :I => (q -> quantile(q, 0.15)) => :I_q_15, # for 70 % prediction intervals
            :I => (q -> quantile(q, 0.25)) => :I_q_25, 
            :I => (q -> quantile(q, 0.75)) => :I_q_75,
            :I => (q -> quantile(q, 0.85)) => :I_q_85,    
            :I => (q -> quantile(q, 0.9)) => :I_q_90,
            :I => (q -> quantile(q, 0.95)) => :I_q_95, # for 90% prediction intervals
            :I => (q -> quantile(q, 0.975)) => :I_q_97p5, # for 95% prediction intervals
            :I => (q -> quantile(q, 0.995)) => :I_q_99p5, # for 99% prediction intervals
            :R => minimum, :R => mean, :R => maximum,
            :R => (q -> quantile(q, 0.005)) => :R_q_p5, # for 99% prediction intervals
            :R => (q -> quantile(q, 0.025)) => :R_q_2p5, # for 95% prediction intervals
            :R => (q -> quantile(q, 0.05)) => :R_q_5, # for 90% prediction intervals
            :R => (q -> quantile(q, 0.1)) => :R_q_10,
            :R => (q -> quantile(q, 0.15)) => :R_q_15, # for 70 % prediction intervals
            :R => (q -> quantile(q, 0.25)) => :R_q_25, 
            :R => (q -> quantile(q, 0.75)) => :R_q_75,
            :R => (q -> quantile(q, 0.85)) => :R_q_85,    
            :R => (q -> quantile(q, 0.9)) => :R_q_90,
            :R => (q -> quantile(q, 0.95)) => :R_q_95, # for 90% prediction intervals
            :R => (q -> quantile(q, 0.975)) => :R_q_97p5, # for 95% prediction intervals
            :R => (q -> quantile(q, 0.995)) => :R_q_99p5, # for 99% prediction intervals
            :β => minimum, :β => mean, :β => maximum,
            :β => (q -> quantile(q, 0.005)) => :β_q_p5, # for 99% prediction intervals
            :β => (q -> quantile(q, 0.025)) => :β_q_2p5, # for 95% prediction intervals
            :β => (q -> quantile(q, 0.05)) => :β_q_5, # for 90% prediction intervals
            :β => (q -> quantile(q, 0.1)) => :β_q_10,
            :β => (q -> quantile(q, 0.15)) => :β_q_15, # for 70 % prediction intervals
            :β => (q -> quantile(q, 0.25)) => :β_q_25, 
            :β => (q -> quantile(q, 0.75)) => :β_q_75,
            :β => (q -> quantile(q, 0.85)) => :β_q_85,    
            :β => (q -> quantile(q, 0.9)) => :β_q_90,
            :β => (q -> quantile(q, 0.95)) => :β_q_95, # for 90% prediction intervals
            :β => (q -> quantile(q, 0.975)) => :β_q_97p5, # for 95% prediction intervals
            :β => (q -> quantile(q, 0.995)) => :β_q_99p5, # for 99% prediction intervals
        )
    return pred_agg
end

function quadratic_dynamics_aggregation(pred_sub)
    pred_agg = combine(groupby(pred_sub, "t"),
        :x => minimum, :x => mean, :x => maximum,
        :x => (q -> quantile(q, 0.005)) => :x_q_p5, # for 99% prediction intervals
        :x => (q -> quantile(q, 0.025)) => :x_q_2p5, # for 95% prediction intervals
        :x => (q -> quantile(q, 0.05)) => :x_q_5, # for 90% prediction intervals
        :x => (q -> quantile(q, 0.1)) => :x_q_10,
        :x => (q -> quantile(q, 0.15)) => :x_q_15, # for 70 % prediction intervals
        :x => (q -> quantile(q, 0.25)) => :x_q_25, 
        :x => (q -> quantile(q, 0.75)) => :x_q_75,
        :x => (q -> quantile(q, 0.85)) => :x_q_85,    
        :x => (q -> quantile(q, 0.9)) => :x_q_90,
        :x => (q -> quantile(q, 0.95)) => :x_q_95, # for 90% prediction intervals
        :x => (q -> quantile(q, 0.975)) => :x_q_97p5, # for 95% prediction intervals
        :x => (q -> quantile(q, 0.995)) => :x_q_99p5) # for 99% prediction intervals)
    return pred_agg
end

function seir_aggregate_over_data_ids(pred_agg)
    
end

function aggregate_predictions(trajectory_pred, data_id, problem_name)
    """
    aggregation for one specific data_id
    """
    pred_sub = DataFrame()
    try
        pred_sub = trajectory_pred[trajectory_pred.data_id .== data_id, :]
    catch
        pred_sub = trajectory_pred
    end
    pred_agg = DataFrame()
    if occursin("seir", problem_name)
        pred_agg = seir_aggregation(pred_sub)
    else
        pred_agg = quadratic_dynamics_aggregation(pred_sub)
    end
    return pred_agg
end


#========================================================================================================
    general plot functions
========================================================================================================#

function histogramm_with_reference(vertical_line_at, x_name, x_vals, x_color=:steelblue4; save_plot=true, plot_path="")
    hist_fig = let
        f = Figure(fonts = (; regular = "Dejavu"))
        ax = CairoMakie.Axis(f[1, 1], title = "Histogramm of $x_name", xlabel = "$x_name")
        hist = CairoMakie.hist!(ax, x_vals, normalization = :pdf, # bar_labels = :values,
            label_size = 15, strokewidth = 0.5, strokecolor = (:white, 0.5), color=x_color) # color=:steelblue4, darkslategray4
        vline = vlines!(ax, [vertical_line_at]; color=:tomato4)
        Legend(f[1, 2],
            [hist, vline],
            ["distribution", "reference"])
        CairoMakie.ylims!(low=0)
        f
    end
    if save_plot 
        save(joinpath(plot_path,"histogramm_$x_name.png"), hist_fig)
    else
        return hist_fig
    end
end

function density_with_reference(vertical_line_at, x_name, x_vals, x_color=:steelblue4; save_plot=true, plot_path="")
    density_fig = let
        f = Figure(fonts = (; regular = "Dejavu"))
        ax = CairoMakie.Axis(f[1, 1], title = "Distribution of $x_name", xlabel = "$x_name")
        hist = CairoMakie.density!(ax, x_vals, normalization = :pdf, 
            label_size = 15, strokecolor = (:white, 0.5), color=x_color) 
        vline = vlines!(ax, [vertical_line_at]; color=:tomato4)
        Legend(f[1, 2],
            [hist, vline],
            ["distribution", "reference"])
        CairoMakie.ylims!(low=0)
        f
    end
    if save_plot 
        save(joinpath(plot_path,"density_$x_name.png"), density_fig)
    else
        return density_fig
    end
end

function density_wo_reference(x_name, x_vals, x_color=:steelblue4; save_plot=true, plot_path="")
    density_fig = let
        f = Figure(fonts = (; regular = "Dejavu"))
        ax = CairoMakie.Axis(f[1, 1], title = "Distribution of $x_name", xlabel = "$x_name")
        hist = CairoMakie.density!(ax, x_vals, normalization = :pdf, 
            label_size = 15, strokecolor = (:white, 0.5), color=x_color) 
        CairoMakie.ylims!(low=0)
        f
    end
    if save_plot 
        save(joinpath(plot_path,"density_$x_name.png"), density_fig)
    else
        return density_fig
    end
end

function r2_plot(col_names, df_stats, plot_path)
    isdir(joinpath(plot_path, "R2")) || mkpath(joinpath(plot_path, "R2"))

    for col in col_names
        R²_plot = let 
            df_sub = dropmissing(df_stats, :negLL_obs_p_opt)
            ys = sort(df_sub[!, "R2_"*col], rev=true)
            # ys = sort(ys)[1:300]
            xs = 1:length(ys)
            f = Figure(fonts = (; regular = "Dejavu"))
            ax = CairoMakie.Axis(f[1, 1], title = col, ylabel="R²")
            scatters = scatterlines!(xs, ys, color = :black, markercolor = ys)
            f
        end
        save(joinpath(plot_path,"R2", "R2_$col.png"), R²_plot)

        R²_plot = let 
            df_sub = dropmissing(df_stats, :negLL_obs_p_opt)
            ys = sort(df_sub[!, "R2_"*col], rev=true)
            # ys = sort(ys)[1:300]
            xs = 1:length(ys)
            f = Figure(fonts = (; regular = "Dejavu"))
            ax = CairoMakie.Axis(f[1, 1], title = col, ylabel="R²")
            scatters = scatterlines!(xs, ys, color = :black, markercolor = ys)
            CairoMakie.ylims!(ax, 0, 1)
            f
        end
        save(joinpath(plot_path,"R2", "R2_cut_$col.png"), R²_plot)
    end
end
