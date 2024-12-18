# Ensemble Settings
n_models = 10000   # number of models to be trained

# arrays = n_models/models_per_cpu

#========================================================================================================
    Cluster Settings
========================================================================================================#
models_per_cpu = 500

#========================================================================================================
    Data Settings
========================================================================================================#
p_ode = [0.9, 0.1, 1000.] # α, γ, N 
u0 = [p_ode[end]*0.995, p_ode[end]*0.004, p_ode[end]*0.001, 0.0]
tspan = (0.0, 130.0)

# Ensemble Settings
subsampling_method = "none" # one of "bootstrap", "repeated_holdout", "resimulate" or "none
if subsampling_method in ["bootstrap", "repeated_holdout"]
    subsample_fraction = 0.8
end

# Data Generation Settings
noise_par = 1.2 # noise level for data generation (std for Gaussian, var/mean ratio for negbin)
noise_model = "negBin" # "Gaussian" or "negBin"
n_timepoints = 30  # indicates how many timepoints are used in the training and validation dataset
n_val = 5 # indicates how many timepoints of n_timepoitns are used in the validation dataset

#========================================================================================================
    UDE Settings
========================================================================================================#
# Ensemble Settings
random_init = true # initialization of parameters

# hyperparameters
hyperpar = (adam_epochs = 4500,
            bfgs_epochs = 0,
            lr_adam = 0.01,
            early_stopping = 5,
            ω = 1e-5,
            act_fct = "tanh",
            hidden_neurons = 6,
            hidden_layers = 2,
            zero_last_layer = true,
            neurons_in = 1,
            neurons_out = 1,
            observe = [3,4]) # states to observe