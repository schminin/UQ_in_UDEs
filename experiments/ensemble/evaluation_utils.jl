#========================================================================================================
    general: summary files
========================================================================================================#

function load_and_evaluate_model(experiment_path, data_id, model_id, hyperpar)
    # load parameters and result

    # p_init = load(joinpath(experiment_path, "models", "dataset_$(data_id)_model_$(model_id)_p_init.jld2"))["p_init"]
    res_opt = load(joinpath(experiment_path, "models", "dataset_$(data_id)_model_$(model_id)_p_opt.jld2"))
    p_opt = res_opt["p_opt"]

    # retrieve mechanistic parameter values
    mech_par = ()
    if problem_name in ["seir", "seir_pulse"]
        mech_par_1 =(
            α_opt = retrieve_α(p_opt["par_α"]),
            γ_opt = retrieve_γ(p_opt["par_γ"]))
        if noise_model == "negBin"
            np_opt = 1 + exp.(p_opt["np"])
        elseif noise_model == "Gaussian"
            np_opt = exp.(p_opt["np"])
        end
        mech_par = ComponentVector(merge(mech_par_1, (np_opt=np_opt, )))
    elseif problem_name=="quadratic_dynamics"
        mech_par_1 =(
            α_opt = retrieve_α(p_opt["par_α"]),
            np_opt = exp.(p_opt["np"]))
        mech_par = ComponentVector(merge(mech_par_1))
    end

    # generate trajectories of model predictions and calculate corresponding metrics
    global nn_model, p_init, st = prepare_model(hyperpar, model_id, sample_mechanistic, noise_model, random_init)
    metrics = []
    metric_names = []
    data_plot = []
    par_list = [(p_opt, "opt")]
    for par in par_list
        # predict
        data = calculate_model_trajectories(data_id, model_id, par[1], train_val_split=true)
        # check whether prediction is successfull
        if size(data.pred_train)==size(data.X_train)  # n_timepoints t_train == n_obs X_train
            # calculate metrics for each prediction
            for (f, f_name) in [(base_loss, "negLL"), (mse, "MSE")]
                for data in [(data.pred_train, data.X_train, "train"), (data.pred_val, data.X_val, "val"), (data.pred_obs, data.X_obs, "obs")]    
                    m = f(data[1][hyperpar.observe,:], data[2][hyperpar.observe,:], par[1].np)
                    push!(metrics, m)
                    push!(metric_names, "$(f_name)_$(data[3])_p_$(par[2])")
                end
            end
            # predict time-dependent parameters
            if problem_name in ["seir", "seir_pulse"]
                β_pred = retrieve_β.(nn_model(data.t_plot', par[1].nn, st)[1][1,:])
                push!(data_plot, (par[2], data.t_plot, data.pred_plot, β_pred))
            elseif problem_name=="quadratic_dynamics"
                push!(data_plot, (par[2], data.t_plot, data.pred_plot))
            end            
        else
            train_success = false
        end
    end

    # results used for model_stats.csv
    col_names = ["data_id", "model_id", metric_names..., labels(mech_par)..., "L2reg_p_opt"]
    col_entry = [data_id, model_id, metrics..., getdata(mech_par)..., hyperpar.ω*norm(p_opt.nn)]

    # results used for model_trajectory.csv
    data_plot
    return col_names, col_entry, data_plot
end

function tidy_prediction_data(data_id, model_id, data)
    # Store predictions in Data Frame
    res = DataFrame()
    # add time column
    res[!, "t"] = data[1][2]

    # for each parameter to consider
    for pred_data in data
        if problem_name in ["seir", "seir_pulse"]
            pred_states = pred_data[3]
            pred_parameter = pred_data[4]
            pred = []
            add_names = []
            try
                pred = [pred_states' pred_parameter]
                add_names = ["$(name)_$(pred_data[1])" for name in [state_names... time_dependent_params]]
            catch
                # n_obs pred_states != n_obs pred_parameter
                continue
            end
        else
            pred = pred_data[3]'
            add_names = ["$(name)_$(pred_data[1])" for name in state_names]
        end

        for elements in zip(1:size(pred)[2], add_names)
            res[!, elements[2]] = pred[:, elements[1]]
        end
    end
    res[!, "data_id"] = repeat([data_id], size(res)[1])
    res[!, "model_id"] = repeat([model_id], size(res)[1])
    return res
end

#========================================================================================================
    general: metrics
========================================================================================================#
function regularization_loss(p_nn, reg_strength)
    reg_strength * norm(p_nn)
end

function R_square(y_pred, y_true)
    μ = mean(y_true)
    # 1 - residual sum of squares / total sum of variances
    R² = 1-sum((y_true.-y_pred).^2)/sum((y_true.-μ).^2)
    return R²
end
#========================================================================================================
    general: ensemble stats
========================================================================================================#

function report_fails(df_stats, n_models)
    n_fails = nrow(filter(:negLL_obs_p_end => x -> !(ismissing(x) || isnothing(x) || isnan(x)), df_stats))
    println("Runs failed in the end: $(n_models-n_fails) of $n_models")
    println("Percentage: $(round((1-n_fails/n_models)*100, digits=2)) % \n")

    n_fails = nrow(filter(:negLL_obs_p_opt => x -> !(ismissing(x) || isnothing(x) || isnan(x)), df_stats))
    println("Runs failed at optimum: $(n_models-n_fails) of $n_models")
    println("Percentage: $(round((1-n_fails/n_models)*100, digits=2)) %")
end
