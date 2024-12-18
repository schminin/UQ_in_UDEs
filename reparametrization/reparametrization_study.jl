using CairoMakie, Plots
using ColorSchemes
#========================================================================================================
    alpha, gamma
========================================================================================================#

"""
    based on a parametrized version of α, 
    calculate the corresponding α

    this parametrization ensures that 
        1.  0<α<24
        2.  values close to zero (corresponding to more stable NNs)
            are related to the more likely value region of α
    
    Note
    the latent period (inverse of alpha) could reasonably be anywhere from an hour 
    (e.g., for certain foodborne illnesses) to several years (e.g., certain malaria cases)
"""
function retrieve_α(α_parameterized)
    (tanh(α_parameterized-2) + 1)*12
end

function parametrize_α(α)
    atanh(α/12-1) + 2
end
α_plot = let
    nn_output = -5:0.1:5
    g = Figure(fonts = (; regular = "Dejavu"), fontsize=25)
    ax = CairoMakie.Axis(g[1, 1], 
        title = "α transformation", xlabel = "parameter value", ylabel="α", 
        backgroundcolor=(:white), yticks = [0, 8, 16, 24])
    l1 = lines!(ax, nn_output, retrieve_α.(nn_output), color=:steelblue4, linewidth = 5)
    s1 = CairoMakie.scatter!(ax, [1], [retrieve_α(1)], color=:tomato)
    desc = text!(ax, "α(output=1)= $(round(retrieve_α(1), digits=2))", position=(1+0.3, retrieve_α(1)-0.8), color=:tomato)
    s1 = CairoMakie.scatter!(ax, [-1], [retrieve_α(-1)], color=:tomato)
    desc = text!(ax, "α(output=-1)= $(round(retrieve_α(-1), digits=2))", position=(-1-3.5, retrieve_α(-1)+0.5), color=:tomato)
    g
end
α_plot
save("reparametrization/α.png", α_plot)

"""
    based on a parametrized version of γ, 
    calculate the corresponding γ

    this parametrization ensures that 
        1.  0 < γ < 1
        2.  values close to zero (corresponding to more stable NNs)
            are related to the more likely value region of gamma
    Note
    we assume, that someone is at least infectious for 1 day
"""
function retrieve_γ(γ_parameterized)
    (tanh(γ_parameterized-1.5) + 1)*0.5
end

function parametrize_γ(γ)
    atanh(γ/1.5-1) + 0.5
end

γ_plot = let
    nn_output = -5:0.1:5
    g = Figure(fonts = (; regular = "Dejavu"), fontsize=25)
    ax = CairoMakie.Axis(g[1, 1], 
        title = "γ transformation", xlabel = "parameter value", ylabel="γ", 
        backgroundcolor=(:white), yticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    l1 = lines!(ax, nn_output, retrieve_γ.(nn_output), color=:darkslategray4, linewidth = 5)
    s1 = CairoMakie.scatter!(ax, [1], [retrieve_γ(1)], color=:tomato)
    desc = text!(ax, "γ(output=1)= $(round(retrieve_γ(1), digits=2))", position=(1+0.2, retrieve_γ(1)-0.03), color=:tomato)
    s1 = CairoMakie.scatter!(ax, [-1], [retrieve_γ(-1)], color=:tomato)
    desc = text!(ax, "γ(output=-1)= $(round(retrieve_γ(-1), digits=2))", position=(-1-3.5, retrieve_γ(-1)+0.02), color=:tomato)
    g
end
γ_plot
save("reparametrization/γ.png", γ_plot)
#========================================================================================================
    beta
========================================================================================================#
"""
    based on a parametrized version of β, 
    calculate the corresponding β

    this parametrization ensures that 
        1.  0<β<3
        2.  values close to zero (corresponding to more stable NNs)
            are related to the more likely value region of beta (~0.05 to ~2.4)
    
    Note
    for beta, consider measles as the classic example of a highly transmissible disease
    (R_0 between 12 and 18 and duration of infectiousness of around a week implies beta between 1.7 and 2.6 per day)
"""
function retrieve_β(β_parameterized)
    (tanh(β_parameterized-1.5) + 1)*1.5
end

function parametrize_β(β)
    atanh(β/1.5-1) + 1.5
end

β_plot = let
    nn_output = -5:0.1:5
    g = Figure(fonts = (; regular = "Dejavu"), linewidth = 5, fontsize=25)
    ax = CairoMakie.Axis(g[1, 1], 
        title = "β transformation", xlabel = "NN output", ylabel="β", 
        backgroundcolor=(:white), yticks = [0, 1, 2, 3, 4, 5])
    l1 = lines!(ax, nn_output, retrieve_β.(nn_output), color=(palette(:default)[5],1.0))
    s1 = CairoMakie.scatter!(ax, [1], [retrieve_β(1)], color=:tomato)
    desc = text!(ax, "β(output=1)= $(round(retrieve_β(1), digits=2))", position=(1+0.21, retrieve_β(1)-0.09), color=:tomato)
    s1 = CairoMakie.scatter!(ax, [-1], [retrieve_β(-1)], color=:tomato)
    desc = text!(ax, "β(output=-1)= $(round(retrieve_β(-1), digits=2))", position=(-1-3.5, retrieve_β(-1)+ 0.09), color=:tomato)
    
    #Legend(g[1,2],
    #    [l1], # [hist, vline],
    #    ["β"])
    g
end
β_plot
save("reparametrization/β.png", β_plot)
