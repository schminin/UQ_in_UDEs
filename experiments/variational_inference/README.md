### Variational Inference 

The main file to reproduce the experiments is called "experiment.jl".
By specifying the problem name ("seir" (= SEIR Waves), "seir_pulse" or "quadratic_dynamics") the dynamic equations are selected.
All other experimental settings are set with a "experimental_setting.jl" file in the respective experiment series and experiment name folders.
These files are provided for all of the presented experiments of the paper. 

The result of an experiment contains a jld2 files describing the posterior at certain stages of the optimization problem. 

The evaluation of the experiment is performed alongside the training in "experiment.jl".
