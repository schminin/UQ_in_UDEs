### MCMC-based UQ

The main file to reproduce the experiments is either called "experiment_pigeon.jl" (for the SEIR Waves and the SEIR Pulse problems) or "experiment_pigeon_quadratic_dynamics.jl". 
By specifying the problem name ("seir" (= SEIR Waves), "seir_pulse" or "quadratic_dynamics") the dynamic equations are selected.
All other experimental settings are set with a "experimental_setting.jl" and "pigeons_setting.jl" file in the respective experiment series and experiment name folders.
These files are provided for all of the presented experiments of the paper. 

The result of an experiment are stored in a result folder at the root of the project folder. Hence, for further analysis it is recommended to move it to the respective experiment folder.
It contains the output of a pigeons experiment for different rounds.

The evaluation of the experiment is performed in the script "pigeon/evaluate.jl".

Since we also ran other experiments without parallel tempering, there exist other files that run these experiments (specifically "experiment.jl"). With this file one can for example run a NUTs sampler.
We have provided an examplary "expeirmental_setting.jl" for a NUTS experiment, too.
