function train_model(p_init, hyperpar, tspan, t_train, X_train, t_weights, t_val, X_val, path, model_id, data_id)
    #save(joinpath(path, "dataset_$(data_id)_model_$(model_id)_p_init.jld2"), 
    #        "model_id", model_id,
    #        "p_init", p_init,
    #        )

    function predict(p, IC, tspan, saveat = t_train)  
        _prob = ODEProblem{true}(dynamics!, IC, tspan, p)
        Array(solve(_prob, Tsit5(), abstol = 1e-8, reltol = 1e-8, saveat = saveat))
    end
    
    function loss(θ, t_obs, X_obs; observe)
        _prob = ODEProblem{true}(dynamics!, IC, (t_obs[1], t_obs[end]), θ)
        tmp_sol = solve(_prob, saveat = t_obs, verbose=false, save_everystep=false)[observe,:]
        if size(tmp_sol) == size(X_obs[observe,:])
            return convert(eltype(θ), base_loss(Array(tmp_sol), X_obs[observe,:], θ.np))
        else
            println(size(tmp_sol))
            println(size(X_obs))
            return convert(eltype(θ), Inf)
        end
    end;
    
    train_loss(p) = loss(p, t_train, X_train; observe=hyperpar.observe) + hyperpar.ω * norm(p.nn)
      
    # assume IC to be known
    IC = X_train[:,1]

    train_losses = []
    val_nlls = []
    function callback(p, l)
        val_loss = base_loss(predict(p, IC, (t_val[1], t_val[end]), t_val)[hyperpar.observe,:], X_val[hyperpar.observe,:], p.np)
        if length(val_nlls)>0
            if (val_loss<=minimum(val_nlls)) 
                save(joinpath(path, "dataset_$(data_id)_model_$(model_id)_p_opt.jld2"),"p_opt", p, "epoch", length(val_nlls))
            end
        end
        push!(val_nlls, val_loss)
        push!(train_losses, l)
        return false
    end
    
    # Set up the optimization problem
    adtype = Optimization.AutoForwardDiff() # negbin can only handle forwarddiff
    optf = OptimizationFunction((p,_) -> train_loss(p), adtype)
    optprob = OptimizationProblem(optf, p_init)
    
    # Hybrid training procedure combining Adam and BFGS
    res1 = solve(optprob, ADAM(hyperpar.lr_adam), callback = callback, maxiters = hyperpar.adam_epochs)
    if hyperpar.bfgs_epochs>0
        optprob2 = OptimizationProblem(optf, res1.u)
        res = solve(optprob2, BFGS(linesearch=Optim.LineSearches.BackTracking()), callback = callback, maxiters = hyperpar.bfgs_epochs)
        solve_time = res1.solve_time + res.solve_time
    else
        res = res1
        solve_time = res.solve_time
    end

    #pred = predict(res.minimizer, IC, (t_train[1], t_train[end]), t_train)
    #pred_val = predict(res.minimizer, IC, (t_val[1], t_val[end]), t_val)

    save(joinpath(path, "dataset_$(data_id)_model_$(model_id)_loss_curves.jld2"), 
            "model_id", model_id,
            "train_losses", train_losses,
            "val_nlls", val_nlls,
            "solve_time", solve_time,
            )
    # print("Final loss: $(res.minimum) \n")
end
