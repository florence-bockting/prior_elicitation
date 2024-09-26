![license](https://raw.githubusercontent.com/florence-bockting/prior_elicitation/badges/.badges/main/poetry-license.svg)
![version](https://raw.githubusercontent.com/florence-bockting/prior_elicitation/badges/.badges/main/poetry-version.svg)
[![Tests](https://github.com/florence-bockting/prior_elicitation/workflows/Tests/badge.svg)](https://github.com/florence-bockting/prior_elicitation/actions)
[![Documentation](https://github.com/florence-bockting/prior_elicitation/workflows/Docs/badge.svg)](https://github.com/florence-bockting/prior_elicitation/actions)
[![Lint](https://github.com/florence-bockting/prior_elicitation/workflows/Linter/badge.svg)](https://github.com/florence-bockting/prior_elicitation/actions?query=workflow%3Alinting)

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
## Authors and Contributors
You are very welcome to contribute to our project. If you find an issue or have a feature request, please use our issue templates.
For those of you who would like to contribute to our project, please have a look at our [contributing guidelines](CONTRIBUTING.md).

**Authors**
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/florence-bockting"><img src="https://avatars.githubusercontent.com/u/48919471?v=4" width="100px;" alt="Florence Bockting"/><br /><sub><b>Florence Bockting</b></sub></a><br /></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/paul-buerkner"><img src="https://avatars.githubusercontent.com/u/12938496?v=4" width="100px;" alt="Paul-Christian Bürkner"/><br /><sub><b>Paul-Christian Bürkner</b></sub></a><br /></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/stefanradev93"><img src="https://avatars.githubusercontent.com/u/22372377?v=4" width="100px;" alt="Stefan T. Radev"/><br /><sub><b>Stefan T. Radev</b></sub></a><br /></td>
    </tr>
  </tbody>
</table>

**Contributors**

<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/bmfazio"><img src="https://avatars.githubusercontent.com/u/26548493?v=4" width="100px;" alt="Luna Fazio"/><br /><sub><b>Luna Fazio</b></sub></a><br /></a><br /><a href="#conceptual-lunafazio" title="Conceptual">🖋</a></td>
    </tr>
  </tbody>
</table>

