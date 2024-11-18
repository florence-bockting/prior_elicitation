
# Supplement for manuscript "Expert elicitation for non-parametric joint prior with normalizing flows"

## Description of folders in directory

+ `plots`: Python scripts with pipeline to replicate diagnostics and plots for simulated data (see `simulation_results` below)
+ `functions`: Internal functions used for computing the diagnostics and creating the plots of the simulation results. 
+ `simulation_scripts`: Python scripts used for running each simulation study.
+ `shell_scripts`: Shell script used for running the Python scripts on a cluster.

The following folders are not included in the GitHub repository (because of size) but need to be created when running the scripts for replicating the results.

+ `experts`: Expert data to train the model. Can be downloaded from OSF [https://osf.io/xrzh6](https://osf.io/xrzh6/)
+ `simulation_results`: Simulation results reported in the manuscript. Can be downloaded from OSF [https://osf.io/xrzh6](https://osf.io/xrzh6/)
+ `sensitivity_results`: Results of sensitivity analysis. Need to be created by running the code in the folder `plots`. Description is provided in comments.


## How to install the prior elicitation method?

+ Install the Python package `elicit` 
    + from the GitHub repository: See "installation" in [README](https://github.com/florence-bockting/prior_elicitation)
    + locally: (1) Clone [GitHub repository](https://github.com/florence-bockting/prior_elicitation) (2) run in cmd `pip install -e .` (ensure that you are in the prior_elicitation folder)

## How to run the simulations studies?

+ *requires*: 
    + Installation of `elicit` package (and its dependencies, see `pyproject.toml`)
    + If you want to use the *expert data* from the manuscript, you can download it from OSF: https://osf.io/xrzh6/. They are included in the `experts.zip`. Download and unpack the zip-folder. Save it in the `manuscript_non_parametric_joint_prior` directory.
+ the code for running the simulation studies can be found in the folder *simulation_scripts*
    + Make sure to adjust the path as needed for the expert data
    
    ```
        expert_data=dict(
            data=pd.read_pickle(
                f"elicit/manuscript_non_parametric_joint_prior/experts/deep_correlated_normal/elicited_statistics.pkl"
            ),
            from_ground_truth=False
        ),
    ```
    
+ the code used for running the scripts on a cluster can be found in *shell scripts*

## How to replicate the plots?

+ *requires*: 
    + Installation of `elicit` package (and its dependencies, see `pyproject.toml`)
    + If your want to use the data from the manuscript you need to download them from OSF: https://osf.io/xrzh6/. Download and unpack both zip-folders (`experts`, `simulation_results`). Save it in the `manuscript_non_parametric_joint_prior` directory.
+ the code for the plots can be found in the folder *plots*
