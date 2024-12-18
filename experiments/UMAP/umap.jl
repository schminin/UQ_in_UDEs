using UMAP: umap
using Plots
using JLD2
using CSV, DataFrames
using Distributions
using ComponentArrays

problem_name = "seir"
experiment_name = "gaussian_IR_noise_0.01"

# load MCMC parameters
parameters_MCMC = load("experiments/MCMC/pigeon/$(problem_name)/$(experiment_name)/parameters.jld2")["parameters"][:,:,1]
parameters_MCMC = Matrix(parameters_MCMC')

# load ensemble parameters
ensemble_path = "experiments/ensemble/ensemble_10000/$(problem_name)/multistart_$(experiment_name)"
stats = CSV.read(joinpath(ensemble_path, "ensemble_stats.csv"), DataFrame)
function get_cutoff(df, p_chi)
    return quantile(Distributions.Chisq(1),1-p_chi)/2 + minimum(skipmissing(df.negLL_obs_p_opt))
end
cutoff_chi_2 = get_cutoff(stats, 0.05)
ensemble_members = stats[stats.negLL_obs_p_opt .<= cutoff_chi_2,"model_id"]

parameters_ensemble = load(joinpath(ensemble_path, "models", "model_$(Int(ensemble_members[1]))_p_opt.jld2"), "p_opt")[1:end]
for i in ensemble_members[2:end]
    parameters_ensemble = hcat(parameters_ensemble, load(joinpath(ensemble_path, "models", "model_$(Int(i))_p_opt.jld2"), "p_opt")[1:end])
end
save("experiments/UMAP/parameters_ensemble.jld2", "parameters_ensemble", parameters_ensemble)

UMAP_ensemble = umap(parameters_ensemble; n_neighbors=10, min_dist=0.1, n_epochs=200)
Plots.scatter(UMAP_ensemble[1,:], UMAP_ensemble[2,:], zcolor=stats[stats.negLL_obs_p_opt .<= cutoff_chi_2,"negLL_obs_p_opt"], 
        title="Ensemble UMAP", label=missing)#, legend=false)# , marker=(2, 2, :auto, stroke(0)))
xlabel!("UMAP 1")
ylabel!("UMAP 2")
savefig("experiments/UMAP/UMAP_ensemble.png")

MCMC_stats = CSV.read("experiments/MCMC/pigeon/$(problem_name)/$(experiment_name)/posterior_stats.csv", DataFrame)
UMAP_MCMC = umap(parameters_MCMC; n_neighbors=10, min_dist=0.01, n_epochs=200)
Plots.scatter(UMAP_MCMC[1,:], UMAP_MCMC[2,:], zcolor = MCMC_stats[:, "negLL_obs"],
        title="MCMC UMAP", label=missing)# , marker=(2, 2, :auto, stroke(0)))
xlabel!("UMAP 1")
ylabel!("UMAP 2")
savefig("experiments/UMAP/UMAP_MCMC.png")

using CairoMakie
using ColorSchemes
parameters_MCMC_trans = copy(parameters_MCMC)
parameters_MCMC_trans[3,:] = log.(sqrt.(exp.(parameters_MCMC_trans[3,:])))# par_np = log(σ²) in MCMC vs par_np = log(σ) in ensemble

parameters_combined = hcat(parameters_ensemble, parameters_MCMC_trans)
parameters_label = vcat(repeat(["ensemble"], size(parameters_ensemble)[2]), repeat(["MCMC"], size(parameters_MCMC_trans)[2]))
UMAP_combined = umap(parameters_combined; n_neighbors=10, min_dist=0.1, n_epochs=200)
Plots.scatter(UMAP_combined[1,:], UMAP_combined[2,:], # zcolor = parameters_label,
        title="UMAP", label=missing)# , marker=(2, 2, :auto, stroke(0)))

Plots.scatter(UMAP_combined[1,1:size(parameters_ensemble)[2]], UMAP_combined[2,1:size(parameters_ensemble)[2]], # zcolor = parameters_label,
        title="UMAP", label=missing)# , marker=(2, 2, :auto, stroke(0)))

fig = let
    f = Figure(size = (1000, 700),  px_per_unit = 10)
    ga = f[1, 1] = GridLayout()
    ax11 = CairoMakie.Axis(ga[1, 1], backgroundcolor=(:white), yticklabelsize=20, xticklabelsize=20, xlabelsize=30, ylabelsize=30, xlabel="UMAP dimension 1", ylabel="UMAP dimension 2") #, width=200)
    s1_legend = CairoMakie.scatter!(ax11, [2], [-2], markersize=15, color=:orange3)
    s1_legend2 = CairoMakie.scatter!(ax11, [2], [-2], markersize=15, color=:dodgerblue4)
    s1_ensemble = CairoMakie.scatter!(ax11,UMAP_combined[1,1:size(parameters_ensemble)[2]], UMAP_combined[2,1:size(parameters_ensemble)[2]] , markersize=7, color=:dodgerblue4)
    s1_MCMC = CairoMakie.scatter!(ax11,UMAP_combined[1,size(parameters_ensemble)[2]+1:end], UMAP_combined[2,size(parameters_ensemble)[2]+1:end] , markersize=7, color=:orange3)
    legend25 = Legend(ga[1,2],
            [s1_legend, s1_legend2], # [hist, vline],
            ["ensemble", "MCMC"], orientation = :vertical, framevisible=false, labelsize=25)
    f
end
save(joinpath("paper_figures", "plots", "UMAP_combined.pdf"), fig)

