# Method for learning prior distributions based on expert knowledge

## Description
This prior elicitation method allows to learn prior distributions of model parameters in a Bayesian model. 
There are two learning methods available: 

+ `parametric prior`: This method allows for learning hyperparameters of pre-specified prior distribution families for each model parameter using batch stochastic gradient descent.
+ `deep prior`: This method allows for learning a joint prior on all model parameters simulateously using normalizing flows.

## Installation
Using `prior_elicitation` as the base directory the package can be installed in development mode as follows:

```
pip install -e .
```

## Usage 
Multiple case studies that demonstrate usage of this method are implemented in the directory `elicit\user_input\configuration_files`

+ open one case study implementation, e.g., `binom_deep.py` and run the file
+ results are saved in a newly created directory `results\data` and `results\plots`

## Notes
This project is under development. Documentation and in-depth examples will follow.