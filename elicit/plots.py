# noqa SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# noqa SPDX-License-Identifier: Apache-2.0

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
import itertools

from typing import Tuple


def initialization(eliobj, cols: int = 4, **kwargs) -> None:
    """
    plots the ecdf of the initialization distribution per hyperparameter

    Parameters
    ----------
    eliobj : instance of :func:`elicit.elicit.Elicit`
        fitted ``eliobj`` object.
    cols : int, optional
        number of columns for arranging the subplots in the figure.
        The default is ``4``.
    **kwargs : any, optional
        additional keyword arguments that can be passed to specify
        `plt.subplots() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html>`_

    Examples
    --------
    >>> el.plots.initialization(eliobj, cols=6)

    >>> el.plots.initialization(eliobj, cols=4, figsize=(8,3))

    """  # noqa: E501
    # get number of hyperparameter
    n_par = len(eliobj.results["init_matrix"].keys())
    # make sure that user uses only as many columns as hyperparameter
    # such that session does not crash...
    if cols > n_par:
        cols = n_par
        print(f"INFO: Reset cols={cols} (total number of hyperparameters)")
    # compute number of rows for subplots
    rows, remainder = np.divmod(n_par, cols)
    # get lower and upper boundary of initialization distr. (x-axis)
    low = tf.subtract(
        eliobj.initializer["distribution"]["mean"],
        eliobj.initializer["distribution"]["radius"],
    )
    high = tf.add(
        eliobj.initializer["distribution"]["mean"],
        eliobj.initializer["distribution"]["radius"],
    )
    if type(low) is not list:
        low = [low] * n_par
    if type(high) is not list:
        high = [high] * n_par
    # use remainder to track which plots should be turned-off/hidden
    if remainder != 0:
        rows += 1
        k = cols - remainder
    else:
        k = remainder

    # plot ecdf of initialiaztion distribution
    # differentiate between subplots that have (1) only one row vs.
    # (2) subplots with multiple rows
    fig, axs = plt.subplots(rows, cols, constrained_layout=True, sharey=True,
                            **kwargs)
    if rows == 1:
        for c, hyp, lo, hi in zip(
            tf.range(cols), eliobj.results["init_matrix"], low, high
        ):
            sns.ecdfplot(
                tf.squeeze(eliobj.results["init_matrix"][hyp]),
                ax=axs[c],
                color="black",
                lw=2,
            )
            axs[c].set_title(f"{hyp}", fontsize="small")
            axs[c].axline((lo, 0), (hi, 1), color="grey", linestyle="dashed",
                          lw=1)
            axs[c].grid(color="lightgrey", linestyle="dotted", linewidth=1)
            axs[c].spines[["right", "top"]].set_visible(False)
            axs[c].tick_params(axis="y", labelsize="x-small")
            axs[c].tick_params(axis="x", labelsize="x-small")
        for k_idx in range(k):
            axs[cols - k_idx - 1].set_axis_off()
    else:
        for (r, c), hyp, lo, hi in zip(
            itertools.product(tf.range(rows), tf.range(cols)),
            eliobj.results["init_matrix"],
            low,
            high,
        ):
            sns.ecdfplot(
                tf.squeeze(eliobj.results["init_matrix"][hyp]),
                ax=axs[r, c],
                color="black",
                lw=2,
            )
            axs[r, c].set_title(f"{hyp}", fontsize="small")
            axs[r, c].axline((lo, 0), (hi, 1), color="grey",
                             linestyle="dashed", lw=1)
            axs[r, c].grid(color="lightgrey", linestyle="dotted", linewidth=1)
            axs[r, c].spines[["right", "top"]].set_visible(False)
            axs[r, c].tick_params(axis="y", labelsize="x-small")
            axs[r, c].tick_params(axis="x", labelsize="x-small")
        for k_idx in range(k):
            axs[rows - 1, cols - k_idx - 1].set_axis_off()
    fig.suptitle("ecdf of initialization distributions", fontsize="medium")
    plt.show()


def loss(eliobj, **kwargs) -> None:
    """
    plots the total loss and the loss per component.

    Parameters
    ----------
    eliobj : instance of :func:`elicit.elicit.Elicit`
        fitted ``eliobj`` object.
    **kwargs : any, optional
        additional keyword arguments that can be passed to specify
        `plt.subplots() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html>`_

    Examples
    --------
    >>> el.plots.loss(eliobj, figsize=(8,3))

    """  # noqa: E501
    try:
        eliobj.history["loss_component"]
    except KeyError:
        print(
            "No information about 'loss_component' found in 'eliobj.history'."
            + "Have you excluded 'loss_components' from history savings?"
        )
    try:
        eliobj.results["elicited_statistics"]
    except KeyError:
        print(
            "No information about 'elicited_statistics' found in "
            + "'eliobj.results'. Have you excluded 'elicited_statistics' from"
            + "results savings?"
        )

    # preprocess loss_component results
    indiv_losses = tf.stack(eliobj.history["loss_component"])
    # names of loss_components
    names_losses = eliobj.results["elicited_statistics"].keys()

    fig, axs = plt.subplots(1, 2, constrained_layout=True, sharex=True,
                            **kwargs)
    axs[0].plot(eliobj.history["loss"], color="black", lw=2)
    for i, name in enumerate(names_losses):
        axs[1].plot(indiv_losses[:, i], label=name, lw=2)
        axs[1].legend(fontsize="small", handlelength=0.4)
    [
        axs[i].set_title(t, fontsize="small")
        for i, t in enumerate(["total loss", "individual losses"])
    ]
    for i in range(2):
        axs[i].set_xlabel("epochs", fontsize="small")
        axs[i].grid(color="lightgrey", linestyle="dotted", linewidth=1)
        axs[i].spines[["right", "top"]].set_visible(False)
        axs[i].tick_params(axis="y", labelsize="x-small")
        axs[i].tick_params(axis="x", labelsize="x-small")


def hyperparameter(eliobj, cols: int = 4, span: int = 30, **kwargs) -> None:
    """
    plots the convergence of each hyperparameter across epochs.

    Parameters
    ----------
    eliobj : instance of :func:`elicit.elicit.Elicit`
        fitted ``eliobj`` object.
    cols : int, optional
        number of columns for arranging the subplots in the figure.
        The default is ``4``.
    span : int, optional
        number of last epochs used to get a final averaged hyperparameter
        value. The default is ``30``.
    **kwargs : any, optional
        additional keyword arguments that can be passed to specify
        `plt.subplots() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html>`_

    Examples
    --------
    >>> el.plots.hyperparameter(eliobj, figuresize=(8,3))

    """  # noqa: E501
    # get number of hyperparameter
    n_par = len(eliobj.history["hyperparameter"].keys())
    # make sure that user uses only as many columns as hyperparameter
    # such that session does not crash...
    if cols > n_par:
        cols = n_par
        print(f"INFO: Reset cols={cols} (total number of hyperparameters)")
    # compute number of rows for subplots
    rows, remainder = np.divmod(n_par, cols)
    # use remainder to track which plots should be turned-off/hidden
    if remainder != 0:
        rows += 1
        k = cols - remainder
    else:
        k = remainder

    fig, axs = plt.subplots(rows, cols, constrained_layout=True, **kwargs)
    if rows == 1:
        for c, hyp in zip(tf.range(cols), eliobj.history["hyperparameter"]):
            # compute mean of last c hyperparameter values
            avg_hyp = tf.reduce_mean(
                eliobj.history["hyperparameter"][hyp][-span:])
            axs[c].axhline(avg_hyp.numpy(), color="darkgrey",
                           linestyle="dotted")
            # plot convergence
            axs[c].plot(eliobj.history["hyperparameter"][hyp], color="black",
                        lw=2)
            axs[c].set_title(f"{hyp}", fontsize="small")
            axs[c].tick_params(axis="y", labelsize="x-small")
            axs[c].tick_params(axis="x", labelsize="x-small")
            axs[c].set_xlabel("epochs", fontsize="small")
            axs[c].grid(color="lightgrey", linestyle="dotted", linewidth=1)
            axs[c].spines[["right", "top"]].set_visible(False)
        for k_idx in range(k):
            axs[cols - k_idx - 1].set_axis_off()
    else:
        for (r, c), hyp in zip(
            itertools.product(tf.range(rows), tf.range(cols)),
            eliobj.history["hyperparameter"],
        ):
            # compute mean of last c hyperparameter values
            avg_hyp = tf.reduce_mean(
                eliobj.history["hyperparameter"][hyp][-span:])
            # plot convergence
            axs[r, c].axhline(avg_hyp.numpy(), color="darkgrey",
                              linestyle="dotted")
            axs[r, c].plot(eliobj.history["hyperparameter"][hyp],
                           color="black", lw=2)
            axs[r, c].set_title(f"{hyp}", fontsize="small")
            axs[r, c].tick_params(axis="y", labelsize="x-small")
            axs[r, c].tick_params(axis="x", labelsize="x-small")
            axs[r, c].set_xlabel("epochs", fontsize="small")
            axs[r, c].grid(color="lightgrey", linestyle="dotted", linewidth=1)
            axs[r, c].spines[["right", "top"]].set_visible(False)
        for k_idx in range(k):
            axs[rows - 1, cols - k_idx - 1].set_axis_off()
    fig.suptitle("Convergence of hyperparameter", fontsize="medium")
    plt.show()


def priors(eliobj, constraints: dict or None = None, **kwargs) -> None:
    """
    plot learned prior distributions of each model parameter based on prior
    samples from last epoch.

    Parameters
    ----------
    eliobj : instance of :func:`elicit.elicit.Elicit`
        fitted ``eliobj`` object.
    constraints : dict or None
        constraints that apply to the model parameters. *Keys* refer to the
        name of the model parameter that should be constraint. *Values* refer
        to the constraint. Currently only 'positive' as constraint is supported.
        Set the argument to None if no constraints should be specified.
        The default value is ``None``.
    **kwargs : any, optional
        additional keyword arguments that can be passed to specify
        `plt.subplots() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html>`_

    Examples
    --------
    >>> el.plots.priors(eliobj, constraints=dict(sigma="positive"),
    >>>                 figsize=(4,4))

    Raises
    ------
    ValueError
        Currently only 'positive' can be used as constraint. Found unsupported
        constraint type.

    """  # noqa: E501
    # check whether constraint specifications are valid
    if constraints is not None:
        for v in list(constraints.values()):
            if (v is not None) and (v != "positive"):
                raise ValueError(
                    "Currently only 'positive' can be used as constraint."
                    +f" Found unsupported constraint type: '{v}'")

    # get parameter names
    name_params = [eliobj.parameters[i]["name"] for
                   i in range(len(eliobj.parameters))]
    # postprocess constraints (add None to all parameters without constraint)
    if constraints is not None:
        for n in name_params:
            if n not in list(constraints.keys()):
                constraints[n]=None
    # get shape of prior samples
    B, n_samples, n_params = eliobj.results["prior_samples"].shape
    # reshape samples by merging batches and number of samples
    priors = tf.reshape(eliobj.results["prior_samples"], (B * n_samples,
                                                          n_params))

    fig, axs = plt.subplots(n_params, n_params, constrained_layout=True,
                            **kwargs)
    for i in range(n_params):
        if constraints is not None:
            if constraints[name_params[i]] == "positive":
                prior = tf.abs(priors[:, i])
            else:
                prior = priors[:, i]
        else:
            prior = priors[:, i]

        sns.kdeplot(prior, ax=axs[i, i], color="black", lw=2)
        axs[i, i].set_xlabel(name_params[i], size="small")
        [axs[i, i].tick_params(axis=a, labelsize="x-small") for
         a in ["x", "y"]]
        axs[i, i].grid(color="lightgrey", linestyle="dotted", linewidth=1)
        axs[i, i].spines[["right", "top"]].set_visible(False)

    for i, j in itertools.combinations(range(n_params), 2):
        if constraints is not None:
            if constraints[name_params[i]] == "positive":
                prior = tf.abs(priors[:, i])
            else:
                prior = priors[:, i]
        else:
            prior = priors[:, i]

        sns.kdeplot(prior, ax=axs[i, i], color="black", lw=2)
        axs[i, j].plot(priors[:, i], priors[:, j], ",", color="black",
                       alpha=0.1)
        [axs[i, j].tick_params(axis=a, labelsize=7) for a in ["x", "y"]]
        axs[j, i].set_axis_off()
        axs[i, j].grid(color="lightgrey", linestyle="dotted", linewidth=1)
        axs[i, j].spines[["right", "top"]].set_visible(False)
    fig.suptitle("Learned prior distributions", fontsize="medium")
    plt.show()


def elicits(eliobj, cols: int = 4, **kwargs) -> None:
    """
    plots the expert-elicited vs. model-simulated statistics.

    Parameters
    ----------
    eliobj : instance of :func:`elicit.elicit.Elicit`
        fitted ``eliobj`` object.
    cols : int, optional
        number of columns for arranging the subplots in the figure.
        The default is ``4``.
    **kwargs : any, optional
        additional keyword arguments that can be passed to specify
        `plt.subplots() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html>`_

    Examples
    --------
    >>> el.plots.elicits(eliobj, cols=4, figsize=(7,3))

    """  # noqa: E501
    def quantiles(
        axs: plt.axes.Axes, expert: tf.Tensor, training: tf.Tensor
    ) -> Tuple[plt.axes.Axes]:
        return (
            axs.axline((0, 0), slope=1, color="darkgrey", linestyle="dashed",
                       lw=1),
            axs.plot(
                expert[0, :], tf.reduce_mean(training, axis=0), "o", ms=5,
                color="black"
            ),
        )

    def correlation(
        axs: plt.axes.Axes, expert: tf.Tensor, training: tf.Tensor
    ) -> Tuple[plt.axes.Axes]:
        return (
            axs.plot(0, expert[:, 0], "s", color="black", label="expert"),
            axs.plot(
                0, tf.reduce_mean(training[:, 0]), "^", color="lightgrey",
                label="train"
            ),
            [
                axs.plot(i, expert[:, i], "s", color="black")
                for i in range(1, training.shape[-1])
            ],
            [
                axs.plot(i, tf.reduce_mean(training[:, i]), "^",
                         color="lightgrey")
                for i in range(1, training.shape[-1])
            ],
            [axs.set_ylim(-1, 1) for i in range(1, training.shape[-1])],
            axs.set_xlim(-0.5, training.shape[-1]),
            axs.set_xticks(
                [i for i in range(training.shape[-1])],
                [f"cor{i}" for i in range(training.shape[-1])],
            ),
            axs.legend(fontsize="x-small", markerscale=0.5),
        )

    # get number of hyperparameter
    n_elicits = len(eliobj.results["expert_elicited_statistics"].keys())
    # make sure that user uses only as many columns as hyperparameter
    # such that session does not crash...
    if cols > n_elicits:
        cols = n_elicits
        print(f"INFO: Reset cols={cols} (total number of elicited statistics)")
    # compute number of rows for subplots
    rows, remainder = np.divmod(n_elicits, cols)
    # use remainder to track which plots should be turned-off/hidden
    if remainder != 0:
        rows += 1
        k = cols - remainder
    else:
        k = remainder

    names_elicits = list(eliobj.results["expert_elicited_statistics"].keys())
    method = [names_elicits[i].split("_")[0] for
              i in range(len(names_elicits))]

    fig, axs = plt.subplots(rows, cols, constrained_layout=True, **kwargs)
    if rows == 1:
        for c, (elicit, meth) in enumerate(zip(names_elicits, method)):
            if meth == "quantiles":
                method = quantiles
            if meth == "pearson":
                method = correlation
            method(
                axs[c],
                eliobj.results["expert_elicited_statistics"][elicit],
                eliobj.results["elicited_statistics"][elicit],
            )
            axs[c].set_title(elicit, fontsize="small")
            axs[c].grid(color="lightgrey", linestyle="dotted", linewidth=1)
            axs[c].spines[["right", "top"]].set_visible(False)
            axs[c].tick_params(axis="y", labelsize="x-small")
            axs[c].tick_params(axis="x", labelsize="x-small")
        for k_idx in range(k):
            axs[cols - k_idx - 1].set_axis_off()
    else:
        for (r, c), elicit, meth in zip(
            itertools.product(tf.range(rows), tf.range(cols)), names_elicits,
            method
        ):
            if meth == "quantiles":
                method = quantiles
            if meth == "pearson":
                method = correlation
            method(
                axs[r, c],
                eliobj.results["expert_elicited_statistics"][elicit],
                eliobj.results["elicited_statistics"][elicit],
            )
            axs[r, c].set_title(elicit, fontsize="small")
            axs[r, c].grid(color="lightgrey", linestyle="dotted", linewidth=1)
            axs[r, c].spines[["right", "top"]].set_visible(False)
            axs[r, c].tick_params(axis="y", labelsize="x-small")
            axs[r, c].tick_params(axis="x", labelsize="x-small")
        for k_idx in range(k):
            axs[rows - 1, cols - k_idx - 1].set_axis_off()
    fig.suptitle("Expert vs. model-simulated elicited statistics",
                 fontsize="medium")
    plt.show()


def marginals(eliobj, cols: int = 4, span: int = 30, **kwargs) -> None:
    """
    plots convergence of mean and sd of the prior marginals

    eliobj : instance of :func:`elicit.elicit.Elicit`
        fitted ``eliobj`` object.
    cols : int, optional
        number of columns for arranging the subplots in the figure.
        The default is ``4``.
    span : int, optional
        number of last epochs used to get a final averaged value for mean and
        sd of the prior marginal. The default is ``30``.
    **kwargs : any, optional
        additional keyword arguments that can be passed to specify
        `plt.subplots() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html>`_

    Examples
    --------
    >>> el.plots.marginals(eliobj, figuresize=(8,3))

    """  # noqa: E501
    def convergence_plot(
        subfigs: plt.Figure.subfigures,
        elicits: tf.Tensor,
        span: int,
        cols: int,
        label: str,
    ) -> plt.Figure.subplots:
        # get number of hyperparameter
        n_par = elicits.shape[-1]
        # make sure that user uses only as many columns as hyperparameter
        # such that session does not crash...
        if cols > n_par:
            cols = n_par
            print(f"INFO: Reset cols={cols} (total number of hyperparameters)")
        # compute number of rows for subplots
        rows, remainder = np.divmod(n_par, cols)
        # use remainder to track which plots should be turned-off/hidden
        if remainder != 0:
            rows += 1
            k = cols - remainder
        else:
            k = remainder

        axs = subfigs.subplots(rows, cols)
        if rows == 1:
            for c, n_hyp in zip(tf.range(cols), tf.range(n_par)):
                # compute mean of last c hyperparameter values
                avg_hyp = tf.reduce_mean(elicits[-span:, n_hyp])
                axs[c].axhline(avg_hyp.numpy(), color="darkgrey",
                               linestyle="dotted")
                # plot convergence
                axs[c].plot(elicits[:, n_hyp], color="black", lw=2)
                axs[c].set_title(rf"{label}($\theta_{n_hyp}$)",
                                 fontsize="small")
                axs[c].tick_params(axis="y", labelsize="x-small")
                axs[c].tick_params(axis="x", labelsize="x-small")
                axs[c].set_xlabel("epochs", fontsize="small")
                axs[c].grid(color="lightgrey", linestyle="dotted", linewidth=1)
                axs[c].spines[["right", "top"]].set_visible(False)
            for k_idx in range(k):
                axs[cols - k_idx - 1].set_axis_off()
        else:
            for (r, c), n_hyp in zip(
                itertools.product(tf.range(rows), tf.range(cols)),
                tf.range(n_par)
            ):
                # compute mean of last c hyperparameter values
                avg_hyp = tf.reduce_mean(elicits[-span:, n_hyp])
                # plot convergence
                axs[r, c].axhline(avg_hyp.numpy(), color="darkgrey",
                                  linestyle="dotted")
                axs[r, c].plot(elicits[:, n_hyp], color="black", lw=2)
                axs[r, c].set_title(rf"$\theta_{n_hyp}$", fontsize="small")
                axs[r, c].tick_params(axis="y", labelsize="x-small")
                axs[r, c].tick_params(axis="x", labelsize="x-small")
                axs[r, c].set_xlabel("epochs", fontsize="small")
                axs[r, c].grid(color="lightgrey", linestyle="dotted",
                               linewidth=1)
                axs[r, c].spines[["right", "top"]].set_visible(False)
            for k_idx in range(k):
                axs[rows - 1, cols - k_idx - 1].set_axis_off()
        return axs

    elicits_means = tf.stack(eliobj.history["hyperparameter"]["means"])
    elicits_std = tf.stack(eliobj.history["hyperparameter"]["stds"])

    fig = plt.figure(layout="constrained", **kwargs)
    subfigs = fig.subfigures(2, 1, wspace=0.07)

    convergence_plot(subfigs[0], elicits_means, span=span, cols=cols,
                     label="mean")
    convergence_plot(subfigs[1], elicits_std, span=span, cols=cols, label="sd")

    fig.suptitle("Convergence of prior marginals mean and sd",
                 fontsize="medium")
    plt.show()
