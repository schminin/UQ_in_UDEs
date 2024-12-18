# MCMC settings
n_chains = 5
n_samples = 100000
samples_per_subchain = 1000

#========================================================================================================
    Data Settings
========================================================================================================#
p_ode = [0.33, 0.05, 1.] # α, γ, N 
u0 = [p_ode[end]*0.995, p_ode[end]*0.004, p_ode[end]*0.001, 0.0]
tspan = (0.0, 130.0)

# Data creation setting
subsampling_method = "none" 

# Data Generation Settings
noise_par = p_ode[end]*0.01   # noise level for data generation
noise_model = "Gaussian" # "Gaussian" or "negBin"
n_timepoints = 30  # indicates how many timepoints are used in the training and validation dataset
n_val = 5 # indicates how many timepoints of n_timepoitns are used in the validation dataset

#========================================================================================================
    UDE Settings
========================================================================================================#
# hyperparameters
hyperpar = (
            act_fct = "tanh",
            hidden_neurons = 6,
            hidden_layers = 2,
            zero_last_layer = true,
            neurons_in = 1,
            neurons_out = 1,
            observe = [3,4]) # states to observe
