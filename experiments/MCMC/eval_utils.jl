function evaluate_parameter_sample(sample_id, par)
    nn_model, p_init, st = prepare_model(hyperpar, 1, sample_mechanistic, noise_model, true, component_vector=false)
    t_obs, X_obs, t_weights = sample_data(1)

    data = calculate_model_trajectories(1, 1, par, train_val_split=false, random_init=true)    
    # results used for model_stats.csv
    metrics = []
    metric_names = []
    for (f, f_name) in [(base_loss, "negLL"), (mse, "MSE")]
        for d in [(data.pred_obs, data.X_obs, "obs")]    
            m = f(d[1][hyperpar.observe,:], d[2][hyperpar.observe,:], par.np)
            push!(metrics, m)
            push!(metric_names, "$(f_name)_$(d[3])")
        end
    end
    df_stats = DataFrame([sample_id, metrics..., norm(par.nn)]', Symbol.(["sample_id", metric_names..., "L2_norm"]))
    
    # results used for model_trajectory.csv
    pred = Float64[]
    add_names = String[]
    if problem_name in ["seir", "seir_pulse"]
        β_pred = retrieve_β.(nn_model(data.t_plot', par.nn, st)[1][1,:])
        pred = [data.pred_plot' β_pred]
        add_names = ["$(name)" for name in [state_names... time_dependent_params]]
    elseif problem_name=="quadratic_dynamics"
        data_plot = (data.t_plot, data.pred_plot)
        pred = data.pred_plot'
        add_names = ["$(name)" for name in state_names]
    end
    df_trajectory = DataFrame()
    df_trajectory[!, "sample_id"] = repeat([sample_id], length(data.t_plot))
    df_trajectory[!, "t"] = data.t_plot
    for elements in zip(1:size(pred)[2], add_names)
        df_trajectory[!, elements[2]] = pred[:, elements[1]]
    end
    return df_stats, df_trajectory
end

function create_posterior_summary_files(experiment_path, mcmc_res, n_trajectories, p_init)
    posterior_samples = sample(mcmc_res, n_trajectories; replace=false)
    df_stats = DataFrame()
    df_trajectory = DataFrame()
    for (sample_id, p) in enumerate(eachrow(Array(posterior_samples)))
        par = merge((par_α = p[1], par_γ=p[2], np=sqrt.(exp.(p[3]))), (nn=vector_to_parameters(p[4:end], p_init.nn),))
        df_stats_sub, df_trajectory_sub = evaluate_parameter_sample(sample_id, par)
        append!(df_stats, df_stats_sub, cols=:union)
        append!(df_trajectory, df_trajectory_sub, cols=:union)
    end
    CSV.write(joinpath(experiment_path, "posterior_stats.csv"), df_stats)
    CSV.write(joinpath(experiment_path, "posterior_trajectory.csv"), df_trajectory)
end

function trace_plot(parameter_name, parameters, parameter_position, retrieve_function, iterations, chain_nrs)
    f = Figure(fonts = (; regular = "Dejavu"))
    ax = CairoMakie.Axis(f[1, 1], title = "Trajectory plot of $parameter_name", xlabel = "iteration", ylabel=parameter_name)
    lines = [CairoMakie.lines!(ax, iterations, retrieve_function.(parameters[iterations, parameter_position, i])) for i in chain_nrs]
    Legend(f[1, 2],
        lines,
        ["chain $i" for i in chain_nrs])
    f
    save(joinpath(experiment_path, "plots", "traceplot", "trace_$parameter_name.png"), f)
end


function constant_parameter_summary(mcmc_chain)
    fig = let
        f = Figure(size = (1000, 300))
        ga = f[1, 1] = GridLayout()

        # Distribution of alpha
        ax1 = CairoMakie.Axis(ga[1, 1], title = "α", backgroundcolor=(:white), xlabelsize=20)
        dens1 = CairoMakie.density!(ax1, retrieve_α.(vcat(Array(get(mcmc_chain, :par_α).par_α)...)), normalization = :pdf, 
                label_size = 15, strokecolor = (:white, 0.5), color=(RGB(0.21, 0.39, 0.55), 0.7)) 
        vline = vlines!(ax1, [p_ode[1]]; color=:tomato4, linewidth=2)
        CairoMakie.ylims!(low=0)
        ax2 = CairoMakie.Axis(ga[1, 2], title = "γ", backgroundcolor=(:white), xlabelsize=20)
        dens2 = CairoMakie.density!(ax2, retrieve_γ.(vcat(Array(get(mcmc_chain, :par_γ).par_γ)...)), normalization = :pdf, 
                label_size = 15, strokecolor = (:white, 0.5), color=(RGB(0.32, 0.55, 0.55), 0.7)) 
        vline = vlines!(ax2, [p_ode[2]]; color=:tomato4, linewidth=2)
        CairoMakie.ylims!(low=0)
        if noise_model=="Gaussian"
            global par_name = "σ"
        elseif noise_model=="negBin"
            global par_name = "overdispersion parameter"
        end
        ax3 = CairoMakie.Axis(ga[1, 3], title = par_name, backgroundcolor=(:white), xlabelsize=20)
        dens3 = CairoMakie.density!(ax3, sqrt.(exp.(vcat(Array(get(mcmc_chain, :par_np).par_np)...))), normalization = :pdf, 
                label_size = 15, strokecolor = (:white, 0.5), color=(RGB(0.68, 0.85, 0.9), 0.7)) 
        vline = vlines!(ax3, [noise_par]; color=:tomato4, linewidth=2)
        CairoMakie.ylims!(low=0)

        Legend(ga[1, 4],
            [[dens1, dens2, dens3], vline],
            ["distribution", "reference"])
        f
    end
    return fig
end


function plot_summary_overview(pred_agg)
    ref = solve(ODEProblem(seir!, u0, tspan, p_ode), saveat = 0.5)

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
        b11 = band!(ax11, pred_agg.t, pred_agg.S_minimum, pred_agg.S_maximum, alpha=0.0, color=(custom_colors[1],0.5))
        l11 = lines!(ax11, pred_agg.t, pred_agg.S_mean, color=(custom_colors[1],1.0), linewidth=3)
        r11 = lines!(ax11, ref.t, Array(ref)[1,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

        b12 = band!(ax12, pred_agg.t, pred_agg.E_minimum, pred_agg.E_maximum, alpha=0.01, color=(custom_colors[2],0.5)) 
        l12 = lines!(ax12, pred_agg.t, pred_agg.E_mean, color=(custom_colors[2],1.0), linewidth=3)
        r12 = lines!(ax12, ref.t, Array(ref)[2,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

        b13 = band!(ax13, pred_agg.t, pred_agg.I_minimum, pred_agg.I_maximum, alpha=0.01, color=(custom_colors[3],0.5)) 
        l13 = lines!(ax13, pred_agg.t, pred_agg.I_mean, color=(custom_colors[3],1.0), linewidth=3)
        s13 = CairoMakie.scatter!(ax13, t_sim, X_sim[3,:], color = (custom_colors[3]-RGB(0.3,0.3,0.3),1), markersize=5)
        r13 = lines!(ax13, ref.t, Array(ref)[3,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

        b14 = band!(ax14, pred_agg.t, pred_agg.R_minimum, pred_agg.R_maximum, alpha=0.01, color=(custom_colors[4],0.5)) 
        l14 = lines!(ax14, pred_agg.t, pred_agg.R_mean, color=(custom_colors[4],1.0), linewidth=3)
        s14 = CairoMakie.scatter!(ax14, t_sim, X_sim[4,:], color = (custom_colors[4]-RGB(0.3,0.3,0.3),1), markersize=5)
        r14 = lines!(ax14, ref.t, Array(ref)[4,:], color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

        legend15 = Legend(ga[1, 6],
            [[r11, r12, r13, r14]], # [hist, vline],
            ["reference"], orientation = :vertical, framevisible=false)

        # Define content of second line
        b21 = band!(ax21, pred_agg.t, pred_agg.S_minimum, pred_agg.S_maximum, alpha=0.0, color=(custom_colors[1],0.5))
        #l21 = lines!(ax21, traj_comb.t, traj_comb.S_opt_mean, color=(custom_colors[1],1.0), linewidth=3)
        #gtb21 = band!(ax21, traj_comb_ground_truth.t, traj_comb_ground_truth.S_opt_minimumimum, traj_comb_ground_truth.S_opt_maximumimum, alpha=0.0, color=(:grey,0.5))
        #gtl21 = lines!(ax21, traj_comb_ground_truth.t, traj_comb_ground_truth.S_opt_mean, color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

        b22 = band!(ax22, pred_agg.t, pred_agg.E_minimum, pred_agg.E_maximum, alpha=0.01, color=(custom_colors[2],0.5)) 
        #l22 = lines!(ax22, traj_comb.t, traj_comb.E_opt_mean, color=(custom_colors[2],1.0), linewidth=3)
        #gtb22 = band!(ax22, traj_comb_ground_truth.t, traj_comb_ground_truth.E_opt_minimumimum, traj_comb_ground_truth.E_opt_maximumimum, alpha=0.0, color=(:grey,0.5))
        #gtl22 = lines!(ax22, traj_comb_ground_truth.t, traj_comb_ground_truth.E_opt_mean, color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

        b23 = band!(ax23, pred_agg.t, pred_agg.I_minimum, pred_agg.I_maximum, alpha=0.01, color=(custom_colors[3],0.5)) 
        #l23 = lines!(ax23, traj_comb.t, traj_comb.I_opt_mean, color=(custom_colors[3],1.0), linewidth=3)
        #s23 = CairoMakie.scatter!(ax23, t_sim, X_sim[3,:], color = (custom_colors[3]-RGB(0.3,0.3,0.3),1), markersize=5)
        #gtb23 = band!(ax23, traj_comb_ground_truth.t, traj_comb_ground_truth.I_opt_minimumimum, traj_comb_ground_truth.I_opt_maximumimum, alpha=0.0, color=(:grey,0.5))
        #gtl23 = lines!(ax23, traj_comb_ground_truth.t, traj_comb_ground_truth.I_opt_mean, color=RGB(0.4,0.4,0.4), linewidth=2, linestyle=:dash)

        b24 = band!(ax24, pred_agg.t, pred_agg.R_minimum, pred_agg.R_maximum, alpha=0.01, color=(custom_colors[4],0.5)) 
        #s24 = CairoMakie.scatter!(ax24, t_sim, X_sim[4,:], color = (custom_colors[4]-RGB(0.3,0.3,0.3),1), markersize=5)
        #l24 = lines!(ax24, traj_comb.t, traj_comb.R_opt_mean, color=(custom_colors[4],1.0), linewidth=3)
        #gtb24 = band!(ax24, traj_comb_ground_truth.t, traj_comb_ground_truth.R_opt_minimumimum, traj_comb_ground_truth.R_opt_maximumimum, alpha=0.0, color=(:grey,0.5))
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
end

function load_chain(chain_nr, experiment_path)
    t = []

    for i in 1:100
        t = vcat(t, load(joinpath(experiment_path, "transitions", "chain_$(chain_nr)_part_$i.jld2"), "transitions"))
    end
    s = load(joinpath(experiment_path, "transitions", "chain_$(chain_nr)_part_100.jld2"), "state")
    # Finally, if you want to convert the vector of `transitions` into a
    # `MCMCChains.Chains` like is typically done:
    chain = AbstractMCMC.bundle_samples(
        map(identity, t),  # trick to concretize the eltype of `transitions`
        model,
        spl,
        s,
        MCMCChains.Chains
    )
    return chain
end
