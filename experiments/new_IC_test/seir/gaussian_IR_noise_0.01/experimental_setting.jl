#========================================================================================================
    Data Settings
========================================================================================================#
p_ode = [0.33, 0.05, 1.] # α, γ, N 
u0 = [p_ode[end]*0.8, p_ode[end]*0.1, p_ode[end]*0.0, p_ode[end]*0.1]
tspan = (0.0, 130.0)
subsampling_method="none"

# Data Generation Settings
noise_par = p_ode[end]*0.01   # noise level for data generation
noise_model = "Gaussian" # "Gaussian" or "negBin"
n_timepoints = 30  # indicates how many timepoints are used in the training and validation dataset
n_val = 5 # indicates how many timepoints of n_timepoitns are used in the validation dataset
