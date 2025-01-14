# noqa SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# noqa SPDX-License-Identifier: Apache-2.0

import tensorflow as tf
import tensorflow_probability as tfp
import bayesflow as bf

tfd = tfp.distributions
bfn = bf.networks


def compute_loss_components(elicited_statistics):
    """
    Computes the single loss components used for computing the discrepancy
    between the elicited statistics. This computation depends on the
    method as specified in the 'combine-loss' argument.

    Parameters
    ----------
    elicited_statistics : dict
        dictionary including the elicited statistics.
    glob_dict : dict
        dictionary including all user-input settings.
    expert : bool
        if workflow is run to simulate a pre-specified ground truth; expert is
        set as 'True'. As consequence the files are saved in a special 'expert'
        folder.

    Returns
    -------
    loss_comp_res : dict
        dictionary including all loss components which will be used to compute
        the discrepancy.

    """

    # extract names from elicited statistics
    name_elicits = list(elicited_statistics.keys())


    # prepare dictionary for storing results
    loss_comp_res = dict()
    # initialize some helpers for keeping track of target quantity
    target_control = []
    i_target = 0
    eval_target = True
    # loop over elicited statistics
    for i, name in enumerate(name_elicits):
        # get name of target quantity
        target = name.split(sep="_")[-1]
        if i != 0:
            # check whether elicited statistic correspond to same target
            # quantity
            eval_target = target_control[-1] == target
        # append current target quantity
        target_control.append(target)
        # if target quantity changes go with index one up
        if not eval_target:
            i_target += 1
        # extract loss component
        loss_comp = elicited_statistics[name]

        assert tf.rank(loss_comp) <= 2, "elicited statistics can only have 2 dimensions."  # noqa

        if tf.rank(loss_comp) == 1:
            # add a last axis for loss computation
            final_loss_comp = tf.expand_dims(loss_comp, axis=-1)
            # store result
            loss_comp_res[f"{name}_loss"] = final_loss_comp
        else:
            loss_comp_res[f"{name}_loss_{i_target}"] = loss_comp

    return loss_comp_res


def compute_discrepancy(loss_components_expert, loss_components_training,
                        targets):
    """
    Computes the discrepancy between all loss components using a specified
    discrepancy measure and returns a list with all loss values.

    Parameters
    ----------
    loss_components_expert : dict
        dictionary including all loss components derived from the
        expert-elicited statistics.
    loss_components_training : dict
        dictionary including all loss components derived from the model
        simulations. (The names (keys) between loss_components_expert and \
                      loss_components_training must match)
    glob_dict : dict
        dictionary including all user-input settings.

    Returns
    -------
    loss_per_component : list
        list of loss value for each loss component

    """

    # create dictionary for storing results
    loss_per_component = []
    # extract expert loss components by name
    keys_loss_comps = list(loss_components_expert.keys())
    # compute discrepancy
    for i, name in enumerate(keys_loss_comps):
        # import loss function
        loss_function = targets[i]["loss"]
        # broadcast expert loss to training-shape
        loss_comp_expert = tf.broadcast_to(
            loss_components_expert[name],
            shape=(
                loss_components_training[name].shape[0],
                loss_components_expert[name].shape[1],
            ),
        )
        # compute loss
        loss = loss_function(loss_comp_expert, loss_components_training[name])
        loss_per_component.append(loss)

    return loss_per_component


def compute_loss(
    training_elicited_statistics, expert_elicited_statistics, epoch, targets
):
    """
    Wrapper around the loss computation from elicited statistics to final
    loss value.

    Parameters
    ----------
    training_elicited_statistics : dict
        dictionary containing the expert elicited statistics.
    expert_elicited_statistics : dict
        dictionary containing the model-simulated elicited statistics.
    global_dict : dict
        global dictionary with all user input specifications.
    epoch : int
        epoch .

    Returns
    -------
    total_loss : float
        total loss value.

    """
    # regularization term for preventing degenerated solutions in var
    # collapse to zero used from Manderson and Goudie (2024)
    def regulariser(prior_samples):
        """
        Regularizer term for loss function: minus log sd of each prior
        distribution (priors with larger sds should be prefered)

        Parameters
        ----------
        prior_samples : tf.Tensor
            samples from prior distributions.

        Returns
        -------
        float
            the negative mean log std across all prior distributions.

        """
        log_sd = tf.math.log(tf.math.reduce_std(prior_samples, 1))
        mean_log_sd = tf.reduce_mean(log_sd)
        return -mean_log_sd

    def compute_total_loss(epoch, loss_per_component, targets):
        """
        applies dynamic weight averaging for multi-objective loss function
        if specified. If loss_weighting has been set to None, all weights
        get an equal weight of 1.

        Parameters
        ----------
        epoch : int
            curernt epoch.
        loss_per_component : list of floats
            list of loss values per loss component.
        global_dict : dict
            global dictionary with all user input specifications.

        Returns
        -------
        total_loss : float
            total loss value (either weighted or unweighted).

        """

        # loss_per_component_current = loss_per_component
        # TODO: check whether order of loss_per_component and target quantities
        # is equivalent!
        total_loss=0
        # create subdictionary for better readability
        for i in range(len(targets)):
            total_loss += tf.multiply(
                loss_per_component[i], targets[i]["weight"]
                )

        return total_loss

    loss_components_expert = compute_loss_components(
        expert_elicited_statistics
    )
    loss_components_training = compute_loss_components(
        training_elicited_statistics
    )
    loss_per_component = compute_discrepancy(
        loss_components_expert, loss_components_training, targets
    )
    weighted_total_loss=compute_total_loss(epoch, loss_per_component, targets)

    return (weighted_total_loss, loss_components_expert,
            loss_components_training, loss_per_component)


def L2(loss_component_expert, loss_component_training,
       axis=None, ord="euclidean"):
    """
    Wrapper around tf.norm that computes the norm of the difference between
    two tensors along the specified axis.
    Used for the correlation loss when priors are assumed to be independent

    Parameters
    ----------
    correlation_training : A Tensor.
    axis     : Any or None
        Axis along which to compute the norm of the difference.
        Default is None.
    ord      : int or str
        Order of the norm. Supports 'euclidean' and other norms
        supported by tf.norm. Default is 'euclidean'.
    """
    difference = tf.subtract(loss_component_expert,
                             loss_component_training)
    norm_values = tf.norm(difference, ord=ord, axis=axis)
    return tf.reduce_mean(norm_values)


class MMD2:
    def __init__(self, kernel : str = "energy", sigma : int or None = None,
                 **kwargs):
        """
        Computes the biased, squared maximum mean discrepancy

        Parameters
        ----------
        kernel : str
            kernel type used for computing the MMD such as "gaussian", "energy"
            The default is "energy".
        sigma : int, optional
            Variance parameter used in the gaussian kernel.
            The default is None.
        **kwargs : keyword arguments
            Additional keyword arguments.

        """
        self.kernel_name = kernel
        self.sigma = sigma

    def __call__(self, x, y):
        """
        Computes the biased, squared maximum mean discrepancy of two samples

        Parameters
        ----------
        x : tensor of shape (batch, num_samples)
            preprocessed expert-elicited statistics.
            Preprocessing refers to broadcasting expert data to same shape as
            model-simulated data.
        y : tensor of shape (batch, num_samples)
            model-simulated statistics corresponding to expert-elicited
            statistics

        Returns
        -------
        MMD2_mean : float
            Average biased, squared maximum mean discrepancy between expert-
            elicited and model simulated data.

        """
        # treat samples as column vectors
        x = tf.expand_dims(x, -1)
        y = tf.expand_dims(y, -1)

        # Step 1
        # compute dot product between samples
        xx = tf.matmul(x, x, transpose_b=True)
        xy = tf.matmul(x, y, transpose_b=True)
        yy = tf.matmul(y, y, transpose_b=True)
        
        # compute squared difference
        u_xx = self.diag(xx)[:,:,None] - 2*xx + self.diag(xx)[:,None,:]
        u_xy = self.diag(xx)[:,:,None] - 2*xy + self.diag(yy)[:,None,:]
        u_yy = self.diag(yy)[:,:,None] - 2*yy + self.diag(yy)[:,None,:]

        # apply kernel function to squared difference
        XX = self.kernel(u_xx, self.kernel_name, self.sigma)
        XY = self.kernel(u_xy, self.kernel_name, self.sigma)
        YY = self.kernel(u_yy, self.kernel_name, self.sigma)

        # Step 2
        # compute biased, squared MMD
        MMD2 = tf.reduce_mean(XX, (1,2)) - 2*tf.reduce_mean(XY, (1,2)) + tf.reduce_mean(YY, (1,2))
        MMD2_mean = tf.reduce_mean(MMD2)

        return MMD2_mean

    def clip(self, u):
        u_clipped = tf.clip_by_value(u, clip_value_min=1e-8, 
                                     clip_value_max=int(1e10))
        return u_clipped

    def diag(self, xx):
        diag = tf.experimental.numpy.diagonal(xx, axis1=1, axis2=2)
        return diag

    def kernel(self, u, kernel, sigma):
        if kernel=="energy":
            # clipping for numerical stability reasons
            d=-tf.math.sqrt(self.clip(u))
        if kernel=="gaussian":
            d=tf.exp(-0.5*tf.divide(u, sigma))
        return d