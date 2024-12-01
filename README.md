<!--
SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>

SPDX-License-Identifier: Apache-2.0
-->

[![DOI](https://zenodo.org/badge/663057594.svg)](https://zenodo.org/doi/10.5281/zenodo.13846929)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests](https://github.com/florence-bockting/prior_elicitation/workflows/Tests/badge.svg)](https://github.com/florence-bockting/prior_elicitation/actions)
[![Documentation](https://github.com/florence-bockting/prior_elicitation/workflows/Docs/badge.svg)](https://github.com/florence-bockting/prior_elicitation/actions)

# Learning prior distributions based on expert knowledge (*Expert knowledge elicitation*)
*Note: This project is still in the development stage and not yet tested for practical use.*

## Description
The `prior_elicitation` package provides a simulation-based framework for learning either parametric or non-parametric, as well as independent or join prior distributions for parameters in a Bayesian model based on expert knowledge.

Further information can be found in the corresponding papers:

+ Bockting, F., Radev S. T., & Bürkner P. C. (2024) Expert-elicitation method for non-parametric joint priors using normalizing flows. Preprint at https://arxiv.org/abs/2411.15826
+ Bockting, F., Radev, S. T. & Bürkner, P. C. (2024). Simulation-based prior knowledge elicitation for parametric Bayesian models. Scientific Reports 14, 17330 (2024). https://doi.org/10.1038/s41598-024-68090-7

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
See our [project website](https://florence-bockting.github.io/prior_elicitation/) with tutorials for usage examples. 

## License
This work is licensed under multiple licences:

  + All original source code is licensed under [Apache License 2.0](LICENSES/Apache-2.0.txt).
  + All documentation is licensed under [CC-BY-SA-4.0](LICENSES/CC-BY-4.0.txt).

## Documentation
Documentation for this project can be found on the [project website](https://florence-bockting.github.io/prior_elicitation/).

## Citation and Reference
This work builds on the following references

+ Bockting, F., Radev, S. T., & Bürkner, P. C. (2024). Simulation-based prior knowledge elicitation for parametric Bayesian models. Scientific Reports, 14(1), 17330. ([see PDF](https://www.nature.com/articles/s41598-024-68090-7.pdf))
+ Bockting, F., Radev S. T., & Bürkner P. C. (2024) Expert-elicitation method for non-parametric joint priors using normalizing flows. Preprint at https://arxiv.org/abs/2411.15826

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
  doi={10.1038/s41598-024-68090-7},
  publisher={Nature Publishing Group UK London}
}

@article{bockting2024expert,
  title={Expert-elicitation method for non-parametric joint priors using normalizing flows},
  author={Bockting, Florence and Radev, Stefan T and B{\"u}rkner, Paul-Christian},
  journal={arXiv preprint},
  year={2024},
  doi={https://arxiv.org/abs/2411.15826}
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

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/lunafazio"><img src="https://avatars.githubusercontent.com/u/26548493?v=4" width="100px;" alt="Luna Fazio"/><br /><sub><b>Luna Fazio</b></sub></a><br /></a><br /><a href="#conceptual-lunafazio" title="Conceptual">🖋</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
