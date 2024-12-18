#========================================================================================================
    Data Settings
========================================================================================================#
p_ode = [1., 2.] # α, β
u0 = [0.1]
tspan = (0.0, 10.0)

# Data creation setting
subsampling_method = "none" 

# Data Generation Settings
noise_par = 0.01   # noise level for data generation
noise_model = "Gaussian" # "Gaussian" or "negBin"
n_timepoints = 12  # indicates how many timepoints are used in the training and validation dataset
n_val = 3 # indicates how many timepoints of n_timepoitns are used in the validation dataset

#========================================================================================================
    UDE Settings
========================================================================================================#
# hyperparameters
hyperpar = (
            act_fct = "tanh",
            hidden_neurons = 3,
            hidden_layers = 2,
            zero_last_layer = true,
            neurons_in = 1,
            neurons_out = 1,
            observe = [3,4]) # states to observe


# for initial parameters
distribution_α = LogUniform(1e-1,1e1)  # log(rand(rng_mech, distribution_α))
distribution_np = LogUniform(1e-1,1e1) # log(rand(rng_mech, distribution_np))