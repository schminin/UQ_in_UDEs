# Ensemble Settings
n_models = 10000   # number of models to be trained

#========================================================================================================
    Cluster Settings
========================================================================================================#
models_per_cpu = 500 # arrays = n_models/models_per_cpu

#========================================================================================================
    Data Settings
========================================================================================================#
p_ode = [1., 2.] # α, β
u0 = [0.1]
tspan = (0.0, 10.0)

# Ensemble Settings
subsampling_method = "none" # one of "bootstrap", "repeated_holdout", "resimulate" or "none
if subsampling_method in ["bootstrap", "repeated_holdout"]
    subsample_fraction = 0.8
end

# Data Generation Settings
noise_par = 0.05   # noise level for data generation
noise_model = "Gaussian" # "Gaussian"
n_timepoints = 12  # indicates how many timepoints are used in the training and validation dataset
n_val = 3 # indicates how many timepoints of n_timepoitns are used in the validation dataset

distribution_α = LogUniform(1e-1,1e1)  # log(rand(rng_mech, distribution_α))
distribution_np = LogUniform(1e-1,1e1) # log(rand(rng_mech, distribution_np))

#========================================================================================================
    UDE Settings
========================================================================================================#
# Ensemble Settings
random_init = true # initialization of parameters

# hyperparameters
hyperpar = (adam_epochs = 4000,
            bfgs_epochs = 500,
            lr_adam = 0.01,
            ω = 1e-4,
            act_fct = "tanh",
            hidden_neurons = 3,
            hidden_layers = 2,
            zero_last_layer = true,
            neurons_in = 1,
            neurons_out = 1,
            observe = [1]) # states to observe