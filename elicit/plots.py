# noqa SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# noqa SPDX-License-Identifier: Apache-2.0

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
import itertools
import pandas as pd

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

    Raises
    ------
    KeyError
        Can't find 'init_matrix' in eliobj.results. Have you excluded it from
        saving?

    ValueError
        if `eliobj.results["init_matrix"]` is None: No samples from
        initialization distribution found. This plot function cannot be used
        if initial values were fixed by the user through the `hyperparams`
        argument in :func:`elicit.elicit.initializer`.

    """  # noqa: E501
    eliobj_res, eliobj_hist, parallel, num_reps = _check_parallel(eliobj)
    # get number of hyperparameter
    n_par = len(eliobj_res["init_matrix"].keys())
    # prepare plot axes
    cols, rows, k, low, high = _prep_subplots(eliobj, cols, n_par,
                                              bounderies=True)

    # check that all information can be assessed
    try:
        eliobj_res["init_matrix"]
    except KeyError:
        print("Can't find 'init_matrix' in eliobj.results."
              +" Have you excluded it from saving?")

    if eliobj_res["init_matrix"] is None:
            raise ValueError(
                "No samples from initialization distribution found."+
                " This plot function cannot be used if initial values were"+
                " fixed by the user through the `hyperparams` argument of"+
                " `initializer`.")

    # plot ecdf of initialiaztion distribution
    # differentiate between subplots that have (1) only one row vs.
    # (2) subplots with multiple rows
    fig, axs = plt.subplots(rows, cols, constrained_layout=True, sharey=True,
                            **kwargs)
    if rows == 1:
        for c, hyp, lo, hi in zip(tf.range(cols), eliobj_res["init_matrix"],
                                  low, high):
            if parallel:
                [sns.ecdfplot(
                    tf.squeeze(eliobj.results[j]["init_matrix"][hyp]),
                    ax=axs[c],
                    color= "black",
                    lw=2,
                    alpha=0.5
                ) for j in range(len(eliobj.results))]
            else:
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
            eliobj_res["init_matrix"],
            low,
            high,
        ):
            if parallel:
                [sns.ecdfplot(
                    tf.squeeze(eliobj.results[j]["init_matrix"][hyp]),
                    ax=axs[r, c],
                    color="black",
                    lw=2,
                    alpha=0.5
                ) for j in range(len(eliobj.results))]
            else:
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

    Raises
    ------
    KeyError
        Can't find 'loss_component' in 'eliobj.history'. Have you excluded
        'loss_components' from history savings?

        Can't find 'loss' in 'eliobj.history'. Have you excluded 'loss' from
        history savings?

        Can't find 'elicited_statistics' in 'eliobj.results'. Have you
        excluded 'elicited_statistics' from results savings?

    """  # noqa: E501
    eliobj_res, eliobj_hist, parallel, n_reps = _check_parallel(eliobj)
    # names of loss_components
    names_losses = eliobj_res["elicited_statistics"].keys()
    # check chains that yield NaN
    if parallel:
        fails, success, success_name = _check_NaN(eliobj, n_reps)

    # check that all information can be assessed
    try:
        eliobj_hist["loss_component"]
    except KeyError:
        print(
            "No information about 'loss_component' found in 'eliobj.history'."
            + "Have you excluded 'loss_components' from history savings?"
        )
    try:
        eliobj_hist["loss"]
    except KeyError:
        print(
            "No information about 'loss' found in 'eliobj.history'."
            + "Have you excluded 'loss' from history savings?"
        )
    try:
        eliobj_res["elicited_statistics"]
    except KeyError:
        print(
            "No information about 'elicited_statistics' found in "
            + "'eliobj.results'. Have you excluded 'elicited_statistics' from"
            + "results savings?"
        )

    fig, axs = plt.subplots(1, 2, constrained_layout=True, sharex=True,
                            **kwargs)
    # plot total loss
    if parallel:
        [axs[0].plot(eliobj.history[i]["loss"], color="black",
                     alpha=0.5, lw=2) for i in success]
    else:
        axs[0].plot(eliobj.history["loss"], color="black", lw=2)
    # plot loss per component
    for i, name in enumerate(names_losses):
        if parallel:
            for j in success:
                # preprocess loss_component results
                indiv_losses = tf.stack(eliobj.history[j]["loss_component"])
                if i==0:
                    axs[1].plot(indiv_losses[:, i], label=name, lw=2,
                                alpha=0.5)
                else:
                    axs[1].plot(indiv_losses[:, i], lw=2,
                                alpha=0.5)
        else:
            # preprocess loss_component results
            indiv_losses = tf.stack(eliobj.history["loss_component"])
            
            axs[1].plot(indiv_losses[:, i], label=name, lw=2)
        axs[1].legend(fontsize="small", handlelength=0.4, frameon=False)
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

    Raises
    ------
    KeyError
        Can't find 'hyperparameter' in 'eliobj.history'. Have you excluded
        'hyperparameter' from history savings?

    """  # noqa: E501
    eliobj_res, eliobj_hist, parallel, n_reps = _check_parallel(eliobj)
    # names of hyperparameter
    names_par = list(eliobj_hist["hyperparameter"].keys())
    # get number of hyperparameter
    n_par = len(names_par)
    # check chains that yield NaN
    if parallel:
        fails, success, success_name = _check_NaN(eliobj, n_reps)
    # prepare subplot axes
    cols, rows, k = _prep_subplots(eliobj, cols, n_par, bounderies=False)

    # check that all information can be assessed
    try:
        eliobj_hist["hyperparameter"]
    except KeyError:
        print(
            "No information about 'hyperparameter' found in "
            + "'eliobj.history'. Have you excluded 'hyperparameter' from"
            + "history savings?"
        )

    fig, axs = plt.subplots(rows, cols, constrained_layout=True, **kwargs)
    if rows == 1:
        for c, hyp in zip(tf.range(cols), names_par):
            if parallel:
                # plot convergence
                [axs[c].plot(eliobj.history[i]["hyperparameter"][hyp],
                            color="black", lw=2, alpha=0.5)
                 for i in success]
            else:
                # compute mean of last c hyperparameter values
                avg_hyp = tf.reduce_mean(
                    eliobj.history["hyperparameter"][hyp][-span:])
                axs[c].axhline(avg_hyp.numpy(), color="darkgrey",
                               linestyle="dotted")
                # plot convergence
                axs[c].plot(eliobj.history["hyperparameter"][hyp], 
                            color="black", lw=2)
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
            itertools.product(tf.range(rows), tf.range(cols)), names_par):
            if parallel:
                [axs[r, c].plot(eliobj.history[i]["hyperparameter"][hyp],
                               color="black", lw=2, alpha=0.5)
                 for i in success]
            else:
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


def prior_joint(eliobj, constraints: dict or None = None,
                idx: int or None = None, **kwargs) -> None:
    """
    plot learned prior distributions of each model parameter based on prior
    samples from last epoch. If parallelization has been used, select which
    replication you want to investigate by indexing it through the 'idx'
    argument.

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
    idx : int or None
        only required if parallelization is used for fitting the method.
        Indexes the replications and allows to choose for which replication the
        prior distributions should be shown.
    **kwargs : any, optional
        additional keyword arguments that can be passed to specify
        `plt.subplots() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html>`_

    Examples
    --------
    >>> el.plots.prior_joint(eliobj,
    >>>                      constraints=dict(sigma="positive"),
    >>>                      figsize=(4,4))

    Raises
    ------
    ValueError
        Currently only 'positive' can be used as constraint. Found unsupported
        constraint type.

        If parallelization has been used for fitting, select one specific
        replication (seed) through the 'idx' argument.

        The value for 'idx' is larger than the number of parallelizations.

    KeyError
        Can't find 'prior_samples' in 'eliobj.results'. Have you excluded
        'prior_samples' from results savings?

    """  # noqa: E501
    # prepare title for plot
    title = "Learned joint prior"

    if type(eliobj.results) is list:
        if idx is None:
            raise ValueError("If parallelization has been used for fitting, "+
                             "select one specific replication (seed) through"+
                             " the 'idx' argument.")
        if idx > len(eliobj.results):
            raise ValueError("The value for 'idx' is larger than the number"+
                             " of parallelizations. 'idx' should not exceed"+
                             f" {len(eliobj.results)} but got {idx}.")
        if len(eliobj.history[int(idx)]["loss"]) < eliobj.trainer["epochs"]:
            raise ValueError(
                f"Training failed for seed with index={idx} (loss is NAN)."+
                " No results for plotting available.")
        # select one result set
        eliobj_res = eliobj.results[int(idx)]
        seed = eliobj.results[int(idx)]["seed"]
        title = title+f" (seed={seed})"
    else:
        eliobj_res = eliobj.results

    # check that all information can be assessed
    try:
        eliobj_res["prior_samples"]
    except KeyError:
        print(
            "No information about 'prior_samples' found in "
            + "'eliobj.results'. Have you excluded 'prior_samples' from"
            + "results savings?"
        )

    # get shape of prior samples
    B, n_samples, n_params = eliobj_res["prior_samples"].shape
    # get parameter names
    name_params = [eliobj.parameters[i]["name"] for i in range(n_params)]
    # reshape samples by merging batches and number of samples
    priors = tf.reshape(eliobj_res["prior_samples"], (B * n_samples, n_params))

    fig, axs = plt.subplots(n_params, n_params, constrained_layout=True,
                            **kwargs)
    for i in range(n_params):
        sns.kdeplot(priors[:, i], ax=axs[i, i], color="black", lw=2)
        axs[i, i].set_xlabel(name_params[i], size="small")
        [axs[i, i].tick_params(axis=a, labelsize="x-small") for
         a in ["x", "y"]]
        axs[i, i].grid(color="lightgrey", linestyle="dotted", linewidth=1)
        axs[i, i].spines[["right", "top"]].set_visible(False)

    for i, j in itertools.combinations(range(n_params), 2):
        sns.kdeplot(priors[:, i], ax=axs[i, i], color="black", lw=2)
        axs[i, j].plot(priors[:, i], priors[:, j], ",", color="black",
                       alpha=0.1)
        [axs[i, j].tick_params(axis=a, labelsize=7) for a in ["x", "y"]]
        axs[j, i].set_axis_off()
        axs[i, j].grid(color="lightgrey", linestyle="dotted", linewidth=1)
        axs[i, j].spines[["right", "top"]].set_visible(False)
    fig.suptitle(title, fontsize="medium")
    plt.show()


def prior_marginals(eliobj, cols: int = 4, constraints: dict or None=None,
                    **kwargs) -> None:
    """
    plots the convergence of each hyperparameter across epochs.

    Parameters
    ----------
    eliobj : instance of :func:`elicit.elicit.Elicit`
        fitted ``eliobj`` object.
    cols : int, optional
        number of columns for arranging the subplots in the figure.
        The default is ``4``.
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
    >>> el.plots.prior_marginals(eliobj, figuresize=(8,3))

    >>> el.plots.prior_marginals(eliobj,
    >>>                          constraints=dict(sigma="positive"),
    >>>                          figuresize=(8,3))

    Raises
    ------
    KeyError
        Can't find 'prior_samples' in 'eliobj.results'. Have you excluded
        'prior_samples' from results savings?

    """  # noqa: E501
    eliobj_res, eliobj_hist, parallel, n_reps = _check_parallel(eliobj)
    # check chains that yield NaN
    if parallel:
        fails, success, success_name = _check_NaN(eliobj, n_reps)
    # get shape of prior samples
    B, n_samples, n_par = eliobj_res["prior_samples"].shape
    # get parameter names
    name_params = [eliobj.parameters[i]["name"] for i in range(n_par)]
    # prepare plot axes
    cols, rows, k = _prep_subplots(eliobj, cols, n_par, bounderies=False)

    # check that all information can be assessed
    try:
        eliobj_res["prior_samples"]
    except KeyError:
        print(
            "No information about 'prior_samples' found in "
            + "'eliobj.results'. Have you excluded 'prior_samples' from"
            + "results savings?"
        )

    fig, axs = plt.subplots(rows, cols, constrained_layout=True, **kwargs)
    if rows == 1:
        for c, par in zip(tf.range(cols), name_params):
            if parallel:
                for i in success:
                    # reshape samples by merging batches and number of samples
                    priors = tf.reshape(eliobj.results[i]["prior_samples"],
                                        (B * n_samples, n_par))
                    sns.kdeplot(priors[:,c], ax=axs[c], color="black", lw=2,
                                alpha=0.5)

            else:
                # reshape samples by merging batches and number of samples
                priors = tf.reshape(eliobj.results["prior_samples"],
                                    (B * n_samples, n_par))
                sns.kdeplot(priors[:,c], ax=axs[c], color="black", lw=2)

            axs[c].set_title(f"{par}", fontsize="small")
            axs[c].tick_params(axis="y", labelsize="x-small")
            axs[c].tick_params(axis="x", labelsize="x-small")
            axs[c].set_xlabel(r"$\theta$", fontsize="small")
            axs[c].set_ylabel("density", fontsize="small")
            axs[c].grid(color="lightgrey", linestyle="dotted", linewidth=1)
            axs[c].spines[["right", "top"]].set_visible(False)
        for k_idx in range(k):
            axs[cols - k_idx - 1].set_axis_off()
    else:
        for j, ((r, c), par) in enumerate(zip(
            itertools.product(tf.range(rows), tf.range(cols)), name_params
        )):
            if parallel:
                for i in success:
                    # reshape samples by merging batches and number of samples
                    priors = tf.reshape(eliobj.results[i]["prior_samples"],
                                        (B * n_samples, n_par))
                    sns.kdeplot(priors[:,j], ax=axs[r, c], color="black",
                                lw=2, alpha=0.5)
            else:
                # reshape samples by merging batches and number of samples
                priors = tf.reshape(eliobj.results["prior_samples"],
                                    (B * n_samples, n_par))
                sns.kdeplot(priors[:,j], ax=axs[r, c], color="black", lw=2)

            axs[r, c].set_title(f"{par}", fontsize="small")
            axs[r, c].tick_params(axis="y", labelsize="x-small")
            axs[r, c].tick_params(axis="x", labelsize="x-small")
            axs[r, c].set_xlabel(r"$\theta$", fontsize="small")
            axs[r, c].set_ylabel("density", fontsize="small")
            axs[r, c].grid(color="lightgrey", linestyle="dotted", linewidth=1)
            axs[r, c].spines[["right", "top"]].set_visible(False)
        for k_idx in range(k):
            axs[rows - 1, cols - k_idx - 1].set_axis_off()
    fig.suptitle("Learned marginal priors", fontsize="medium")
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

    Raises
    ------
    KeyError
        Can't find 'expert_elicited_statistics' in 'eliobj.results'. Have you
        excluded 'expert_elicited_statistics' from results savings?

        Can't find 'elicited_statistics' in 'eliobj.results'. Have you
        excluded 'elicited_statistics' from results savings?

    """  # noqa: E501
    # check whether parallelization has been used
    eliobj_res, eliobj_hist, parallel, n_reps = _check_parallel(eliobj)
    # get number of hyperparameter
    n_elicits = len(eliobj_res["expert_elicited_statistics"].keys())
    # check chains that yield NaN
    if parallel:
        fails, success, success_name = _check_NaN(eliobj, n_reps)
    # prepare plot axes
    cols, rows, k = _prep_subplots(eliobj, cols, n_elicits, bounderies=False)
    # extract quantities of interest needed for plotting
    name_elicits = list(eliobj_res["expert_elicited_statistics"].keys())
    method = [name_elicits[i].split("_")[0] for i in range(n_elicits)]

    # check that all information can be assessed
    try:
        eliobj_res["expert_elicited_statistics"]
    except KeyError:
        print(
            "No information about 'expert_elicited_statistics' found in "
            + "'eliobj.results'. Have you excluded 'expert_elicited_statistics'"
            + " from results savings?"
        )
    try:
        eliobj_res["elicited_statistics"]
    except KeyError:
        print(
            "No information about 'elicited_statistics' found in "
            + "'eliobj.results'. Have you excluded 'elicited_statistics'"
            + " from results savings?"
        )

    # plotting
    fig, axs = plt.subplots(rows, cols, constrained_layout=True, **kwargs)
    if rows == 1:
        for c, (elicit, meth) in enumerate(zip(name_elicits, method)):
            if meth == "quantiles":
                labels = [None]*n_reps
                prep = (axs[c].axline((0, 0), slope=1, color="darkgrey",
                                      linestyle="dashed", lw=1),)
                method = _quantiles

            if meth == "cor":
                # prepare labels for plotting
                labels = [("expert","train")]+[
                    (None,None) for i in range(n_reps-1)]
                # select method function
                method = _correlation
                # get number of correlations
                num_cor = eliobj_res["elicited_statistics"][elicit].shape[-1]
                prep = (
                    axs[c].set_ylim(-1, 1),
                    axs[c].set_xlim(-0.5, num_cor),
                    axs[c].set_xticks(
                        [i for i in range(num_cor)],
                        [f"cor{i}" for i in range(num_cor)],
                    ),
                )

            if parallel:
                for i in success:
                    method(
                        axs[c],
                        eliobj.results[i]["expert_elicited_statistics"][elicit],
                        eliobj.results[i]["elicited_statistics"][elicit],
                        labels[i]
                    )+prep
            else:
                method(
                    axs[c],
                    eliobj.results["expert_elicited_statistics"][elicit],
                    eliobj.results["elicited_statistics"][elicit],
                    labels[0]
                )+prep

            if elicit.endswith("_cor"):
                axs[c].legend(fontsize="x-small", markerscale=0.5, 
                              frameon=False)
            axs[c].set_title(elicit, fontsize="small")
            axs[c].grid(color="lightgrey", linestyle="dotted", linewidth=1)
            axs[c].spines[["right", "top"]].set_visible(False)
            axs[c].tick_params(axis="y", labelsize="x-small")
            axs[c].tick_params(axis="x", labelsize="x-small")
            if not elicit.endswith("_cor"):
                axs[c].set_xlabel("expert", fontsize="small")
                axs[c].set_ylabel("model-sim.", fontsize="small")
        for k_idx in range(k):
            axs[cols - k_idx - 1].set_axis_off()

    else:
        for (r, c), elicit, meth in zip(
            itertools.product(tf.range(rows), tf.range(cols)), name_elicits,
            method
        ):
            if meth == "quantiles":
                labels = [None]*n_reps
                prep = (axs[r, c].axline((0, 0), slope=1, color="darkgrey",
                                      linestyle="dashed", lw=1),)
                method = _quantiles

            if meth == "cor":
                labels = [("expert","train")]+[
                    (None,None) for i in range(n_reps-1)]
                method = _correlation
                num_cor = eliobj_res["elicited_statistics"][elicit].shape[-1]
                prep = (
                    axs[r, c].set_ylim(-1, 1),
                    axs[r, c].set_xlim(-0.5, num_cor),
                    axs[r, c].set_xticks(
                        [i for i in range(num_cor)],
                        [f"cor{i}" for i in range(num_cor)],
                    ),
                )

            if parallel:
                for i in success:
                    method(
                        axs[r, c],
                        eliobj.results[i]["expert_elicited_statistics"][elicit],
                        eliobj.results[i]["elicited_statistics"][elicit],
                        labels[i]
                    )+prep
            else:
                method(
                    axs[r, c],
                    eliobj.results["expert_elicited_statistics"][elicit],
                    eliobj.results["elicited_statistics"][elicit],
                    labels[0]
                )+prep

            if elicit.endswith("_cor"):
                axs[r, c].legend(fontsize="x-small", markerscale=0.5,
                                 frameon=False)
            axs[r, c].set_title(elicit, fontsize="small")
            axs[r, c].grid(color="lightgrey", linestyle="dotted", linewidth=1)
            axs[r, c].spines[["right", "top"]].set_visible(False)
            axs[r, c].tick_params(axis="y", labelsize="x-small")
            axs[r, c].tick_params(axis="x", labelsize="x-small")
            if not elicit.endswith("_cor"):
                axs[r, c].set_xlabel("expert", fontsize="small")
                axs[r, c].set_ylabel("model-sim.", fontsize="small")
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
    kwargs : any, optional
        additional keyword arguments that can be passed to specify
        `plt.subplots() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html>`_

    Examples
    --------
    >>> el.plots.marginals(eliobj, figuresize=(8,3))

    Raises
    ------
    KeyError
        Can't find 'hyperparameter' in 'eliobj.history'. Have you excluded 
        'hyperparameter' from history savings?

    """  # noqa: E501

    # check that all information can be assessed
    try:
        eliobj.history["hyperparameter"]
    except KeyError:
        print(
            "No information about 'hyperparameter' found in 'eliobj.history'"
            +" Have you excluded 'hyperparameter' from history savings?"
        )

    elicits_means = tf.stack(eliobj.history["hyperparameter"]["means"])
    elicits_std = tf.stack(eliobj.history["hyperparameter"]["stds"])

    fig = plt.figure(layout="constrained", **kwargs)
    subfigs = fig.subfigures(2, 1, wspace=0.07)
    _convergence_plot(subfigs[0], elicits_means, span=span, cols=cols,
                     label="mean")
    _convergence_plot(subfigs[1], elicits_std, span=span, cols=cols,
                      label="sd")
    fig.suptitle("Convergence of prior marginals mean and sd",
                 fontsize="medium")
    plt.show()


def priorpredictive(eliobj, **kwargs) -> None:
    """
    plots prior predictive distribution of samples from the generative model
    in the last epoch

    Parameters
    ----------
    eliobj : instance of :func:`elicit.elicit.Elicit`
        fitted ``eliobj`` object.
    kwargs : any, optional
        additional keyword arguments that can be passed to specify
        `plt.subplots() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html>`_


    Examples
    --------
    >>> el.plots.priorpredictive(eliobj, figuresize=(6,2))

    Raises
    ------
    KeyError
        Can't find 'target_quantities' in 'eliobj.results'. Have you excluded 
        'target_quantities' from results savings?

    """
    # check that all information can be assessed
    try:
        eliobj.results["target_quantities"]
    except KeyError:
        print(
            "No information about 'target_quantities' found in 'eliobj.results'"
            +" Have you excluded 'target_quantities' from results savings?"
        )

    target_reshaped=[]
    for k in eliobj.results["target_quantities"]:
        target = eliobj.results["target_quantities"][k]
        target_reshaped.append(tf.reshape(target, (target.shape[0]*target.shape[1])))
    
    targets = tf.stack(target_reshaped,-1)

    fig, axs = plt.subplots(1, 1, constrained_layout=True, **kwargs)
    axs.grid(color="lightgrey", linestyle="dotted", linewidth=1)
    for i in range(targets.shape[-1]):
        shade = i / (targets.shape[-1] - 1)
        color = plt.cm.gray(shade)
        sns.histplot(targets[:,i], stat="probability", bins=40,
                     label=eliobj.targets[i]["name"], ax=axs,
                     color=color
                     )
    plt.legend(fontsize="small", handlelength=0.9, frameon=False)
    axs.set_title("prior predictive distribution", fontsize="small")
    axs.spines[["right", "top"]].set_visible(False)
    axs.tick_params(axis="y", labelsize="x-small")
    axs.tick_params(axis="x", labelsize="x-small")
    axs.set_xlabel(r"$y_{pred}$", fontsize="small")
    plt.show()


def prior_averaging(eliobj, cols: int=4, n_sim: int=10_000,
                    height_ratio: list=[1,1.5], weight_factor: float=1.,
                    **kwargs) -> None:
    # prepare plotting
    n_par = len(eliobj.parameters)
    name_par = [eliobj.parameters[i]["name"] for i in range(n_par)]
    n_reps = len(eliobj.results)
    # prepare plot axes
    cols, rows, k = _prep_subplots(eliobj, cols, n_par)
    # remove chains for which training yield NaN
    fail, success, success_name = _check_NaN(eliobj, n_reps)

    # perform model averaging
    w_MMD, averaged_priors, B, n_samples = _model_averaging(
        eliobj, weight_factor, success, n_sim)
    # store results in data frame
    df = pd.DataFrame(dict(weight= w_MMD, seed=success_name))
    # sort data frame according to weight values
    df_sorted=df.sort_values(by="weight", ascending=False
                             ).reset_index(drop=True)

    # plot average and single priors
    fig = plt.figure(layout="constrained", **kwargs)
    subfigs = fig.subfigures(2, 1, height_ratios=height_ratio)
    subfig0 = subfigs[0].subplots(1, 1)
    subfig1 = subfigs[1].subplots(rows, cols)

    # plot weights of model averaging
    sns.barplot(y="seed", x="weight", data=df_sorted, ax=subfig0,
                color="darkgrey", orient="h", order=df_sorted["seed"])
    subfig0.spines[["right", "top"]].set_visible(False)
    subfig0.grid(color="lightgrey", linestyle="dotted", linewidth=1)
    subfig0.set_xlabel("weight", fontsize="small")
    subfig0.set_ylabel("seed", fontsize="small")
    subfig0.tick_params(axis="y", labelsize="x-small")
    subfig0.tick_params(axis="x", labelsize="x-small")
    subfig0.set_xlim(0,df_sorted["weight"][0]+0.1)

    # plot individual priors and averaged prior
    if rows == 1:
        for c, par in zip(tf.range(cols), name_par):
            for i in success:
                # reshape samples by merging batches and number of samples
                prior = tf.reshape(eliobj.results[i]["prior_samples"],
                                    (B * n_samples, n_par))
                sns.kdeplot(prior[:,c], ax=subfig1[c], color="black", lw=2,
                            alpha=0.5)
            avg_prior = tf.reshape(averaged_priors,(B * n_sim, n_par))
            sns.kdeplot(avg_prior[:,c], color="red", ax=subfig1[c],
                        label="average")
            subfig1[c].set_title(f"{par}", fontsize="small")
            subfig1[c].tick_params(axis="y", labelsize="x-small")
            subfig1[c].tick_params(axis="x", labelsize="x-small")
            subfig1[c].set_xlabel(r"$\theta$", fontsize="small")
            subfig1[c].set_ylabel("density", fontsize="small")
            subfig1[c].grid(color="lightgrey", linestyle="dotted",
                            linewidth=1)
            subfig1[c].spines[["right", "top"]].set_visible(False)
            subfig1[c].legend(handlelength=0.3, fontsize="small",
                                 frameon=False)
        for k_idx in range(k):
            subfig1[cols - k_idx - 1].set_axis_off()
    else:
        for j, ((r, c), par) in enumerate(zip(
            itertools.product(tf.range(rows), tf.range(cols)), name_par
        )):
            for i in success:
                # reshape samples by merging batches and number of samples
                priors = tf.reshape(eliobj.results[i]["prior_samples"],
                                    (B * n_samples, n_par))
                sns.kdeplot(priors[:,j], ax=subfig1[r, c], color="black",
                            lw=2, alpha=0.5)
            avg_prior = tf.reshape(averaged_priors,(B * n_sim, n_par))
            sns.kdeplot(avg_prior[:, j], color="red", ax=subfig1[r, c],
                        label="average")
            subfig1[r, c].set_title(f"{par}", fontsize="small")
            subfig1[r, c].tick_params(axis="y", labelsize="x-small")
            subfig1[r, c].tick_params(axis="x", labelsize="x-small")
            subfig1[r, c].set_xlabel(r"$\theta$", fontsize="small")
            subfig1[r, c].set_ylabel("density", fontsize="small")
            subfig1[r, c].grid(color="lightgrey", linestyle="dotted",
                               linewidth=1)
            subfig1[r, c].spines[["right", "top"]].set_visible(False)
            subfig1[r, c].legend(handlelength=0.3, fontsize="small",
                                 frameon=False)
        for k_idx in range(k):
            subfig1[rows - 1, cols - k_idx - 1].set_axis_off()
    subfigs[0].suptitle("Prior averaging (weights)", fontsize="small",
                        ha="left", x=0.)
    subfigs[1].suptitle("Prior distributions", fontsize="small",
                        ha="left", x=0.)
    fig.suptitle("Prior averaging", fontsize="medium")
    plt.show()


def _model_averaging(eliobj, weight_factor, success, n_sim):
    # compute final loss per run by averaging over last x values
    mean_losses = np.stack(
        [np.mean(eliobj.history[i]["loss"]) for i in success]
    )
    # retrieve min MMD
    min_loss = min(mean_losses)
    # compute Delta_i MMD
    delta_MMD = mean_losses - min_loss
    # relative likelihood
    rel_likeli = np.exp(float(weight_factor) * delta_MMD)
    # compute Akaike weights
    w_MMD = rel_likeli / np.sum(rel_likeli)
    
    # model averaging
    # extract prior samples; shape = (num_sims, B*sim_prior, num_param)
    prior_samples = tf.stack([eliobj.results[i]["prior_samples"] for 
                              i in success],0)
    num_success, B, n_samples, n_par = prior_samples.shape

    # sample component
    sampled_component = np.random.choice(
        np.arange(num_success), size=n_sim, replace=True, p=w_MMD
    )
    # sample observation index
    sampled_obs = np.random.choice(np.arange(n_samples), size=n_sim,
                                   replace=True)
    # select prior
    averaged_priors = tf.stack(
        [prior_samples[rep,:, obs, :] for rep, obs in zip(
        sampled_component, sampled_obs)])

    return w_MMD, averaged_priors, B, n_samples


def _check_parallel(eliobj):
    if type(eliobj.results) is list:
        eliobj_res = eliobj.results[0]
        eliobj_hist = eliobj.history[0]
        parallel = True
        num_reps = len(eliobj.results)
    else:
        eliobj_res = eliobj.results
        eliobj_hist = eliobj.history
        parallel = False
        num_reps=1

    return eliobj_res, eliobj_hist, parallel, num_reps


def _quantiles(axs: any, expert: tf.Tensor, training: tf.Tensor,
              labels: None=None) -> Tuple[any]:
    return (
        axs.plot(
            expert[0, :], tf.reduce_mean(training, axis=0), "o", ms=5,
            color="black", alpha=0.5
        ),
    )


def _correlation(axs: any, expert: tf.Tensor, training: tf.Tensor,
                labels: list[tuple]) -> Tuple[any]:
    return (
        axs.plot(0, expert[:, 0], "*", color="red", label=labels[0],
                 zorder=2),
        axs.plot(
            0, tf.reduce_mean(training[:, 0]), "s", color="black",
            label=labels[1], alpha=0.5, zorder=1
        ),
        [
            axs.plot(i, expert[:, i], "*", color="red",
                     zorder=2) for i in range(1, training.shape[-1])
        ],
        [
            axs.plot(i, tf.reduce_mean(training[:, i]), "s",
                     color="black", alpha=0.5, zorder=1)
            for i in range(1, training.shape[-1])
        ],
    )


def _prep_subplots(eliobj, cols, n_quant, bounderies=False):
    # make sure that user uses only as many columns as hyperparameter
    # such that session does not crash...
    if cols > n_quant:
        cols = n_quant
        print(
            f"INFO: Reset cols={cols} (number of elicited statistics)")
    # compute number of rows for subplots
    rows, remainder = np.divmod(n_quant, cols)
    
    if bounderies:
        # get lower and upper boundary of initialization distr. (x-axis)
        low = tf.subtract(
            eliobj.initializer["distribution"]["mean"],
            eliobj.initializer["distribution"]["radius"],
            )
        high = tf.add(
            eliobj.initializer["distribution"]["mean"],
            eliobj.initializer["distribution"]["radius"],
            )
        try:
            len(low)
        except TypeError:
            low = [low] * n_quant
            high = [high] * n_quant
        else:
            pass

    # use remainder to track which plots should be turned-off/hidden
    if remainder != 0:
        rows += 1
        k = cols - remainder
    else:
        k = remainder

    if bounderies:
        return cols, rows, k, low, high
    else:
        return cols, rows, k


def _convergence_plot(
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


def _check_NaN(eliobj, n_reps):
    # check whether some replications stopped with NAN
    ep_run = [len(eliobj.history[i]["loss"]) for i in range(n_reps)]
    seed_rep = [eliobj.results[i]["seed"] for i in range(n_reps)]
    # extract successful and failed seeds and indices for further plotting
    fail=[]
    success=[]
    success_name=[]
    for i, ep in enumerate(ep_run):
        if ep < eliobj.trainer["epochs"]:
            fail.append((i,seed_rep[i]))
        else:
            success.append(i)
            success_name.append(seed_rep[i])
    if len(fail) > 0:
        print(
        f"INFO: {len(fail)} of {n_reps} replications yield loss NAN and are"+
        f" excluded from the plot. Failed seeds: {fail} (index, seed)")
    return fail, success, success_name