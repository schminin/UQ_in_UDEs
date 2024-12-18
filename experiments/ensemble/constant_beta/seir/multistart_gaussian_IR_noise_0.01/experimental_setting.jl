# Ensemble Settings
aleatoric_noise_realizations = 1 # number of independent ensembles 
ensemble_members = 10000 # number of models per ensemble

# arrays = n_models/models_per_cpu

#========================================================================================================
    Cluster Settings
========================================================================================================#
n_cpus = 40

#========================================================================================================
    Data Settings
========================================================================================================#
p_ode = [0.33, 0.05, 1.] # α, γ, N 
u0 = [p_ode[end]*0.995, p_ode[end]*0.004, p_ode[end]*0.001, 0.0]
tspan = (0.0, 130.0)

# Ensemble Settings
subsampling_method = "none" # one of "bootstrap", "repeated_holdout", "resimulate" or "none
if subsampling_method in ["bootstrap", "repeated_holdout"]
    subsample_fraction = 0.8
end

# Data Generation Settings
noise_par = p_ode[end]*0.01   # noise level for data generation
noise_model = "Gaussian" # "Gaussian" or "negBin"
n_timepoints = 30  # indicates how many timepoints are used in the training and validation dataset
n_val = 5 # indicates how many timepoints of n_timepoitns are used in the validation dataset

#========================================================================================================
    UDE Settings
========================================================================================================#
# Ensemble Settings
random_init = true # initialization of parameters

# hyperparameters
hyperpar = (adam_epochs = 4000,
            bfgs_epochs = 1000,
            lr_adam = 0.01,
            observe = [3,4]) # states to observe