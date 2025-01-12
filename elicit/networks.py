# noqa SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# noqa SPDX-License-Identifier: Apache-2.0

def NF(inference_network: callable, network_specs: dict,
       base_distribution: callable):
    """
    specification of the normalizing flow used from BayesFlow library

    Parameters
    ----------
    inference_network : callable
        type of inference network as specified by bayesflow.inference_networks.
    network_specs : dict
        specification of normalizing flow architecture. Arguments are inherited
        from chosen bayesflow.inference_networks.
    base_distribution : callable
        Base distribution from which should be sampled during learning.
        Normally the base distribution is a multivariate normal.
    input_dim : int
        number of model parameters.

    Returns
    -------
    nf_dict : dict
        dictionary specifying the normalizing flow settings.

    """
    nf_dict = dict(
        inference_network=inference_network,
        network_specs=network_specs,
        base_distribution=base_distribution
    )

    return nf_dict