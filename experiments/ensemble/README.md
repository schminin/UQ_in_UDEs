### Ensemble-based Uncertainty Quantification

The main file to reproduce the experiments is called "experiment.jl". 
By specifying the problem name ("seir" (= SEIR Waves), "seir_pulse" or "quadratic_dynamics") the dynamic equations are selected.
All other experimental settings are set with a "experimental_setting.jl" file in the respective experiment series and experiment name folders.
The "experimental_setting.jl" files are provided for all of the presented experiments of the paper. 

The result of an experiment are all potential ensemble members parameters (initial, optimal according to the validation loss and parameters at the end of the optimization routine).
Relevant for the ensemble selection are only the optimal parameters.

Evaluation and the selection of the final ensemble members can be performed using the script "ensemble_evaluation.jl".


Additionally to the basic ensemble-based UQ, we performed an experiment keeping beta constant. All code needed for this experiment can be found in the subfolder "constant_beta".