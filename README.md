# Method for learning prior distributions based on expert knowledge

## Description
This prior elicitation method allows to learn prior distributions of model parameters in a Bayesian model. 
There are two learning methods available: 

+ `parametric prior`: This method allows for learning hyperparameters of pre-specified prior distribution families for each model parameter using batch stochastic gradient descent.
+ `deep prior`: This method allows for learning a joint prior on all model parameters simulateously using normalizing flows.

## Installation

+ requires: Python >=3.10 and <= 3.12

```
# Create an empty Python environment (here with conda example)
conda create -n elicitation-env python=3.11

# activate environment
conda activate elicitation-env

# install elicit package
pip install git+https://github.com/florence-bockting/prior_elicitation.git
```

## Usage 
See the [introductory example](https://florence-bockting.github.io/prior_elicitation/introductory_example.html) for a minimal example with implementation.

## Notes
This project is under development. Documentation and in-depth examples will follow.
