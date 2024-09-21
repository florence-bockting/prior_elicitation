[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

# Learning prior distributions based on expert knowledge (Expert knowledge elicitation method)
*Note: This project is still in the development stage and not yet tested for practical use.*

## Description
A central characteristic of Bayesian statistics is the ability to consistently incorporate prior knowledge into various modeling processes. In this paper, we focus on translating domain expert knowledge into corresponding prior distributions over model parameters, a process known as prior elicitation. Expert knowledge can manifest itself in diverse formats, including information about raw data, summary statistics, or model parameters. A major challenge for existing elicitation methods is how to effectively utilize all of these different formats in order to formulate prior distributions that align with the expert’s expectations, regardless of the model structure. To address these challenges, we develop a simulation-based elicitation method that can learn the hyperparameters of potentially any parametric prior distribution from a wide spectrum of expert knowledge using stochastic gradient descent. We validate the effectiveness and robustness of our elicitation method in four representative case studies covering linear models, generalized linear models, and hierarchical models. Our results support the claim that our method is largely independent of the underlying model structure and adaptable to various elicitation techniques, including quantile-based, moment-based, and histogram-based methods.

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

## License
This work is licensed under a [AGPL v3 license](LICENSE).

## Documentation
Documentation for this project can be found on the [project website](https://florence-bockting.github.io/prior_elicitation/).

## Citation and Reference
This work builds on the following references

Bockting, F., Radev, S. T., & Bürkner, P. C. (2024). Simulation-based prior knowledge elicitation for parametric Bayesian models. Scientific Reports, 14(1), 17330. ([see PDF](https://www.nature.com/articles/s41598-024-68090-7.pdf))

**BibTeX:**
```
@article{bockting2024simulation,
  title={Simulation-based prior knowledge elicitation for parametric Bayesian models},
  author={Bockting, Florence and Radev, Stefan T and B{\"u}rkner, Paul-Christian},
  journal={Scientific Reports},
  volume={14},
  number={1},
  pages={17330},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```

