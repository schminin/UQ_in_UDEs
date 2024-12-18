#========================================================================================================
    quadratic dynamics: plots
========================================================================================================#

function quadratic_dynamics_density_plots(plot_path, df_sub, df_stats)
    isdir(joinpath(plot_path, "distribution")) || mkpath(joinpath(plot_path, "distribution"))
    # α
    #histogramm_with_reference(p_ode[1], "α", df_sub.α_opt, :steelblue4; plot_path=plot_path)
    density_with_reference(p_ode[1], "α optimal", df_sub.α_opt, :steelblue4; plot_path=joinpath(plot_path, "distribution"))
    density_with_reference(p_ode[1], "α init", df_stats.α_init, :steelblue4; plot_path=joinpath(plot_path, "distribution"))

    # noise parameter 
    xs = df_sub.np_opt[df_sub.np_opt.<Inf]
    density_with_reference(noise_par, "σ", xs, :lightblue; plot_path=joinpath(plot_path, "distribution"))
    xs = df_sub.np_init[df_sub.np_init.<Inf]
    density_with_reference(noise_par, "σ init", xs, :lightblue; plot_path=joinpath(plot_path, "distribution"))

    # negLL 
    density_wo_reference("Negative Log Likelihood (cut off)", df_sub.negLL_obs_p_opt, :plum4; plot_path=joinpath(plot_path, "distribution"))
end

function quadratic_dynamics_trajectory_plots(plot_path, df_sub)
    trajectory_plot_path = joinpath(plot_path, "trajectory")
    isdir(trajectory_plot_path) || mkpath(trajectory_plot_path)
    
    # Visualize predictions:
    cutoff_models = df_sub.model_id
    df_traj_sub = df_trajectory[[x in cutoff_models for x in df_trajectory.model_id],:]
    gdf = groupby(df_traj_sub, :t)
    traj_comb = combine(gdf, 
        :x_opt => minimum, :x_opt => mean, :x_opt => maximum)

    custom_colors = palette(:tab10)[1:5]
    trajectory_plot = let
        g = Figure(fonts = (; regular = "Dejavu"))
        ax = CairoMakie.Axis(g[1, 1], title = "Trajectory", xlabel = "time", backgroundcolor=(:white))
        b1 = band!(ax, traj_comb.t, traj_comb.x_opt_minimum, traj_comb.x_opt_maximum, alpha=0.0, color=(custom_colors[5],0.5))
        s1 = CairoMakie.scatter!(ax, t_sim, X_sim[1,:], color = (custom_colors[5]-RGB(0.3,0.3,0.3),1), markersize=6)
        l1 = lines!(ax, traj_comb.t, traj_comb.x_opt_mean, color=(custom_colors[5]-RGB(0.1,0.1,0.1),1))
        legend = Legend(g[1,2],
        [[b1, l1], [s1]], # [hist, vline],
        ["x", "observed"])
        g
    end
    save(joinpath(trajectory_plot_path,"trajectories_w_bands.png"), trajectory_plot)


    # With reference:
    ref = solve(ODEProblem(simulation_dynamics!, u0, tspan, p_ode), saveat = saveat_plot)

    trajectory_plot = let
        g = Figure(fonts = (; regular = "Dejavu"))
        ax = CairoMakie.Axis(g[1, 1], title = "Trajectory", xlabel = "time", backgroundcolor=(:white))
        b1 = band!(ax, traj_comb.t, traj_comb.x_opt_minimum, traj_comb.x_opt_maximum, alpha=0.0, color=(custom_colors[5],0.5))
        s1 = CairoMakie.scatter!(ax, t_sim, X_sim[1,:], color = (custom_colors[5]-RGB(0.3,0.3,0.3),1), markersize=6)
        l1 = lines!(ax, traj_comb.t, traj_comb.x_opt_mean, color=(custom_colors[5]-RGB(0.1,0.1,0.1),1))
        l2 = lines!(ax, ref.t, Array(ref)[1,:], color=:grey)
        legend = Legend(g[1,2],
        [[b1, l1], [s1], l2], # [hist, vline],
        ["x", "observed", "reference"])
        g
    end
    save(joinpath(trajectory_plot_path,"trajectories_w_reference.png"), trajectory_plot)
end

#========================================================================================================
    seir: plots
========================================================================================================#

function seir_distribution_plots(plot_path, df_sub)
    isdir(joinpath(plot_path, "distribution")) || mkpath(joinpath(plot_path, "distribution"))
    # alpha
    #histogramm_with_reference(p_ode[1], "α", df_sub.α_opt, :steelblue4; save_plot=true, plot_path=plot_path)
    density_with_reference(p_ode[1], "α", df_sub.α_opt, :steelblue4; save_plot=true, plot_path=joinpath(plot_path, "distribution"))
   
    # gamma
    #histogramm_with_reference(p_ode[2], "γ", df_sub.γ_opt, :darkslategray4; save_plot=true, plot_path=plot_path)
    density_with_reference(p_ode[2], "γ", df_sub.γ_opt, :darkslategray4; save_plot=true, plot_path=joinpath(plot_path, "distribution"))

    # Distribution of the np
    if noise_model=="Gaussian"
        x_name = "standard deviation"
    elseif noise_model=="negBin"
        x_name = "overdispersion parameter"
    end
    #histogramm_with_reference(noise_par, x_name, df_sub.np_opt, :lightblue; save_plot=true, plot_path=plot_path)
    density_with_reference(noise_par, x_name, df_sub.np_opt, :lightblue; save_plot=true, plot_path=joinpath(plot_path, "distribution"))

    # Distribution of NegLL
    density_wo_reference("Negative Log Likelihood (cut off)", df_sub.negLL_obs_p_opt, :plum4; plot_path=joinpath(plot_path, "distribution"))

    # 2D Distribution of Parameters
    hist_2d_α_γ = let 
        x_vals_α= df_sub.α_opt
        x_vals_γ = df_sub.γ_opt
        p1 =Plots.histogram2d(x_vals_α, x_vals_γ)
        Plots.title!("2D Histogram of mechanistic parameters")
        Plots.xlabel!("α")
        Plots.ylabel!("γ")
        p1
    end
    save(joinpath(plot_path, "distribution", "histogramm2d_α_γ.png"), hist_2d_α_γ)
end

function seir_trajectory_plots(plot_path, df_sub)
    trajectory_plot_path = joinpath(plot_path, "trajectory")
    isdir(trajectory_plot_path) || mkpath(trajectory_plot_path)

    # Visualize predictions:
    cutoff_models = df_sub.model_id
    df_traj_sub = df_trajectory[[x in cutoff_models for x in df_trajectory.model_id],:]
    gdf = groupby(df_traj_sub, :t)
    traj_comb = combine(gdf, 
        :S_opt => minimum, :S_opt => mean, :S_opt => maximum,
        :E_opt => minimum, :E_opt => mean, :E_opt => maximum,
        :I_opt => minimum, :I_opt => mean, :I_opt => maximum,
        :R_opt => minimum, :R_opt => mean, :R_opt => maximum,
        :β_opt => minimum, :β_opt => mean, :β_opt => maximum)

    # state predictions 
    custom_colors = palette(:tab10)[1:4] # get(ColorSchemes.viridis, [i/(n_colors-1) for i in 0:1:(n_colors-1)])
    trajectory_plot = let
        g = Figure(fonts = (; regular = "Dejavu"))
        ax = CairoMakie.Axis(g[1, 1], title = "Trajectories", xlabel = "time", backgroundcolor=(:white))
        b1 = band!(ax, traj_comb.t, traj_comb.S_opt_minimum, traj_comb.S_opt_maximum, alpha=0.0, color=(custom_colors[1],0.5))
        b2 = band!(ax, traj_comb.t, traj_comb.E_opt_minimum, traj_comb.E_opt_maximum, alpha=0.01, color=(custom_colors[2],0.5)) 
        b3 = band!(ax, traj_comb.t, traj_comb.I_opt_minimum, traj_comb.I_opt_maximum, alpha=0.01, color=(custom_colors[3],0.5)) 
        b4 = band!(ax, traj_comb.t, traj_comb.R_opt_minimum, traj_comb.R_opt_maximum, alpha=0.01, color=(custom_colors[4],0.5)) 
        #s1 = CairoMakie.scatter!(ax, t_sim, X_sim[1,:], color = (custom_colors[1],1), markersize=6)
        #s2 = CairoMakie.scatter!(ax, t_sim, X_sim[2,:], color = (custom_colors[2],1), markersize=6)
        s3 = CairoMakie.scatter!(ax, t_sim, X_sim[3,:], color = (custom_colors[3]-RGB(0.3,0.3,0.3),1), markersize=6)
        s4 = CairoMakie.scatter!(ax, t_sim, X_sim[4,:], color = (custom_colors[4]-RGB(0.3,0.3,0.3),1), markersize=6)
        l1 = lines!(ax, traj_comb.t, traj_comb.S_opt_mean, color=(custom_colors[1],1.0))
        l2 = lines!(ax, traj_comb.t, traj_comb.E_opt_mean, color=(custom_colors[2],1.0))
        l3 = lines!(ax, traj_comb.t, traj_comb.I_opt_mean, color=(custom_colors[3],1.0))
        l4 = lines!(ax, traj_comb.t, traj_comb.R_opt_mean, color=(custom_colors[4],1.0))
        legend = Legend(g[1,2],
        [[b1, l1], [b2, l2], [b3, l3], [b4,l4], [s3, s4]], # [hist, vline],
        ["S", "E", "I", "R", "observed"])
        g
    end
    trajectory_plot
    save(joinpath(trajectory_plot_path,"trajectories_w_bands.png"), trajectory_plot)

    trajectory_plot = let
        g = Figure(fonts = (; regular = "Dejavu"))
        ax = CairoMakie.Axis(g[1, 1], title = "Mean Ensemble Trajectories", xlabel = "time", backgroundcolor=(:white))
        l1 = lines!(ax, traj_comb.t, traj_comb.S_opt_mean, color=(custom_colors[1],1.0))
        l2 = lines!(ax, traj_comb.t, traj_comb.E_opt_mean, color=(custom_colors[2],1.0))
        l3 = lines!(ax, traj_comb.t, traj_comb.I_opt_mean, color=(custom_colors[3],1.0))
        l4 = lines!(ax, traj_comb.t, traj_comb.R_opt_mean, color=(custom_colors[4],1.0))
        s3 = CairoMakie.scatter!(ax, t_sim, X_sim[3,:], color = (custom_colors[3]-RGB(0.3,0.3,0.3),1), markersize=6)
        s4 = CairoMakie.scatter!(ax, t_sim, X_sim[4,:], color = (custom_colors[4]-RGB(0.3,0.3,0.3),1), markersize=6)
        Legend(g[1,2],
        [l1, l2, l3, l4, [s3, s4]], # [hist, vline],
        ["S", "E", "I", "R", "observed"])
        g
    end
    save(joinpath(trajectory_plot_path,"trajectories.png"), trajectory_plot)

    # β predictions
    β_plot = let
        g = Figure(fonts = (; regular = "Dejavu"))
        ax = CairoMakie.Axis(g[1, 1], title = "mean ensemble β", xlabel = "time", backgroundcolor=(:white))
        l1 = lines!(ax, traj_comb.t, traj_comb.β_opt_mean, color=(palette(:default)[5],1.0))
        b1 = band!(ax, traj_comb.t, traj_comb.β_opt_minimum, traj_comb.β_opt_maximum, alpha=0.0, color=(palette(:default)[5],0.5))
        t = tspan[1]:1:tspan[end]
        l2 = lines!(ax, tspan[1]:1:tspan[end], β.(t), color=RGB(0.4, 0.4, 0.3), linestyle = :dash)
        Legend(g[1,2],
            [l1, l2], # [hist, vline],
            ["β", "reference"])
        g
    end
    save(joinpath(trajectory_plot_path,"β_trajectories_w_bands.png"), β_plot)

    # 50% of best models below threshold
    fraction = 0.5
    cutoff_models = (sort(df_sub, :negLL_obs_p_opt)[1:Int(round(nrow(df_sub)*fraction)), :]).model_id
    df_traj_sub2 = df_trajectory[[x in cutoff_models for x in df_trajectory.model_id],:]
    gdf2 = groupby(df_traj_sub2, :t)
    traj_comb2 = combine(gdf2, 
        :S_opt => minimum, :S_opt => mean, :S_opt => maximum,
        :E_opt => minimum, :E_opt => mean, :E_opt => maximum,
        :I_opt => minimum, :I_opt => mean, :I_opt => maximum,
        :R_opt => minimum, :R_opt => mean, :R_opt => maximum,
        :β_opt => minimum, :β_opt => mean, :β_opt => maximum)


    β_plot = let
        g = Figure(fonts = (; regular = "Dejavu"))
        ax = CairoMakie.Axis(g[1, 1], title = "β trajectory", xlabel = "time", backgroundcolor=(:white))
        l1 = lines!(ax, traj_comb.t, traj_comb.β_opt_mean, color=(palette(:default)[5],1.0))
        b1 = band!(ax, traj_comb.t, traj_comb.β_opt_minimum, traj_comb.β_opt_maximum, alpha=0.0, color=(palette(:default)[5],0.3))
        b2 = band!(ax, traj_comb2.t, traj_comb2.β_opt_minimum, traj_comb2.β_opt_maximum, alpha=0.0, color=(palette(:default)[5],0.25))
        t = tspan[1]:1:tspan[end]
        l2 = lines!(ax, tspan[1]:1:tspan[end], β.(t), color=RGB(0.4, 0.4, 0.3), linestyle = :dash)
        Legend(g[1,2],
            [l1, l2, b1], # [hist, vline],
            ["β", "reference", "50%/100% of ensembles"])
        g
    end
    save(joinpath(trajectory_plot_path,"β_trajectories_w_multiple_bands.png"), β_plot)
end

function alpha_beta_trajectory(plot_path, df_trajectory, df_stats)
    trajectory_plot_path = joinpath(plot_path, "trajectory")
    isdir(trajectory_plot_path) || mkpath(trajectory_plot_path)
    
    df_comb = rightjoin(df_trajectory[!, ["model_id", "t", "β_opt"]], df_stats[!, ["model_id", "α_opt"]], on=:model_id)
    df_comb[!,"α_times_β"] = df_comb.α_opt .* df_comb.β_opt
    gdf = groupby(df_comb, :t)
    traj_comb = combine(gdf, :α_times_β => minimum, :α_times_β => mean, :α_times_β => maximum)
    α_times_β_plot = let
        g = Figure(fonts = (; regular = "Dejavu"))
        ax = CairoMakie.Axis(g[1, 1], title = "mean ensemble α times β", xlabel = "time", backgroundcolor=(:white))
        l1 = lines!(ax, traj_comb.t, traj_comb.α_times_β_mean)
        b1 = band!(ax, traj_comb.t, traj_comb.α_times_β_minimum, traj_comb.α_times_β_maximum, alpha=0.8)
        t = tspan[1]:1:tspan[end]
        l2 = lines!(ax, tspan[1]:1:tspan[end], β.(t) .* p_ode[1], linestyle = :dash)
        Legend(g[1,2],
            [l1, l2], # [hist, vline],
            ["α times β", "reference"])
        g
    end
    save(joinpath(trajectory_plot_path,"α_times_β_with_bands.png"), α_times_β_plot)

    α_times_β_plot = let
        g = Figure(fonts = (; regular = "Dejavu"))
        ax = CairoMakie.Axis(g[1, 1], title = "mean ensemble α times β", xlabel = "time", backgroundcolor=(:white))
        l1 = lines!(ax, traj_comb.t, traj_comb.α_times_β_mean)
        t = tspan[1]:1:tspan[end]
        l2 = lines!(ax, tspan[1]:1:tspan[end], β.(t) .* p_ode[1], linestyle = :dash)
        Legend(g[1,2],
            [l1, l2], # [hist, vline],
            ["α times β", "reference"])
        g
    end
    save(joinpath(trajectory_plot_path,"α_times_β.png"), α_times_β_plot)
end

#========================================================================================================
    quadratic dynamics: plots
========================================================================================================#

function quadratic_dynamics_train_test_split(t_train, X_train, t_val, X_val)
    sample_plot = let 
        ref = solve(ODEProblem(simulation_dynamics!, u0, tspan, p_ode), saveat = saveat_plot)
        f = Figure(fonts = (; regular = "Dejavu"))
        ax = CairoMakie.Axis(f[1, 1], 
            title = "Simulation Data", ylabel="x", xlabel="t")
        l1 = lines!(ref.t, Array(ref)[1,:], color=:grey)
        s2 = CairoMakie.scatter!(t_val[2:end], X_val[1,2:end], color=:tomato)
        s1 = CairoMakie.scatter!(t_train[2:end], X_train[1,2:end], color=:peru)
        s3 = CairoMakie.scatter!(t_train[1], u0[1], color=:black)
        Legend(f[1, 2],
            [l1, s1, s2, s3],
            ["reference", "training", "validation", "IC"])
        f
    end
    sample_plot
end

function seir_train_test_split(t_train, X_train, t_val, X_val)
    sample_plot = let 
        ref = solve(ODEProblem(simulation_dynamics!, u0, tspan, p_ode), saveat = saveat_plot)
        f = Figure(fonts = (; regular = "Dejavu"))
        ax = CairoMakie.Axis(f[1, 1], 
            title = "Simulation Data", ylabel="", xlabel="t")
        l1 = lines!(ref.t, Array(ref)[1,:], color=(custom_colors[1],1.0))
        l2 = lines!(ref.t, Array(ref)[2,:], color=(custom_colors[2],1.0))
        l3 = lines!(ref.t, Array(ref)[3,:], color=(custom_colors[3],1.0))
        l4 = lines!(ref.t, Array(ref)[4,:], color=(custom_colors[4],1.0))
        s1 = CairoMakie.scatter!(t_train[2:end], X_train[3,2:end], color=:peru)
        s2 = CairoMakie.scatter!(t_train[2:end], X_train[4,2:end], color=:peru)
        s3 = CairoMakie.scatter!(t_val[2:end], X_val[3,2:end], color=:grey)
        s4 = CairoMakie.scatter!(t_val[2:end], X_val[4,2:end], color=:grey)
        ic1 = CairoMakie.scatter!(t_train[1], u0[1], color=:black)
        ic2 = CairoMakie.scatter!(t_train[1], u0[2], color=:black)
        ic3 = CairoMakie.scatter!(t_train[1], u0[3], color=:black)
        ic4 = CairoMakie.scatter!(t_train[1], u0[4], color=:black)
        Legend(f[1, 2],
            [l1, l2, l3, l4, [s1, s2], [s3, s4], [ic1, ic2, ic3, ic4]],
            ["S", "E", "I", "R", "training", "validation", "IC"])
        f
    end
    sample_plot
end