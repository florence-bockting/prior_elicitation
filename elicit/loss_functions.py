# noqa SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# noqa SPDX-License-Identifier: Apache-2.0

import tensorflow as tf


def norm_diff(correlation_training, axis=None, ord="euclidean"):
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
    difference = tf.subtract(tf.zeros(correlation_training.shape),
                             correlation_training)
    norm_values = tf.norm(difference, ord=ord, axis=axis)
    return tf.reduce_mean(norm_values)


class MmdEnergy(tf.Module):
    def __call__(self, loss_component_expert, loss_component_training,
                 **kwargs):
        """
        Computes the maximum-mean discrepancy between two samples.

        Parameters
        ----------
        loss_component_expert : tf.Tensor
            loss component based on expert input.
        loss_component_training : tf.Tensor
            loss component based on model simulations.
        **kwargs : optional additional keyword arguments
            optional: additional keyword arguments.

        Returns
        -------
        mmd_loss : float
            maximum mean discrepancy loss using an energy kernel

        """
        # get batch size
        B = loss_component_expert.shape[0]
        # dimensionality for expert input and model simulations
        dim_expert = loss_component_expert.shape[-1]
        dim_training = loss_component_training.shape[-1]

        # expert samples and model simulation samples
        x = tf.expand_dims(
            tf.reshape(loss_component_expert, (B, dim_expert)),
            -1)
        y = tf.expand_dims(
            tf.reshape(loss_component_training, (B, dim_training)), -1)

        a = self.generate_weights(x, dim_expert)
        b = self.generate_weights(y, dim_training)

        return self.kernel_loss(x, y, a, b, dim_expert)

    def generate_weights(self, loss_component, dim):
        B, dim, _ = loss_component.shape
        weights = tf.divide(tf.ones(shape=(B, dim),
                                    dtype=loss_component.dtype), dim)
        return weights

    # (x*x - 2xy + y*y)
    def squared_distances(self, loss_component_expert,
                          loss_component_training):
        # (B, N, 1)
        distance_expert = tf.expand_dims(
            tf.math.reduce_sum(
                tf.multiply(loss_component_expert, loss_component_expert),
                axis=-1
            ),
            axis=2,
        )
        # (B, 1, M)
        distance_training = tf.expand_dims(
            tf.math.reduce_sum(
                tf.multiply(loss_component_training, loss_component_training),
                axis=-1
            ),
            axis=1,
        )
        # (B, N, M)
        distance_expert_training = tf.matmul(
            loss_component_expert, tf.transpose(loss_component_training,
                                                perm=(0, 2, 1))
        )
        # compute sq. distance
        squared_distance = (
            distance_expert - 2 * distance_expert_training + distance_training
        )
        return squared_distance

    # -sqrt[(x*x - 2xy + y*y)]
    def distances(self, loss_component_expert, loss_component_training):
        distance = tf.math.sqrt(
            tf.clip_by_value(
                self.squared_distances(loss_component_expert,
                                       loss_component_training),
                clip_value_min=1e-8,
                clip_value_max=int(1e10),
            )
        )
        # energy distance as negative distance
        energy_distance = -distance
        return energy_distance

    # helper function
    def scal(self, a, f):
        B = a.shape[0]
        return tf.math.reduce_sum(
            tf.reshape(a, (B, -1)) * tf.reshape(f, (B, -1)), axis=1
        )

    # k(x,y)=0.5*sum_i sum_j a_i a_j k(x_i,x_j) -
    # sum_i sum_j a_i b_j k(x_i,y_j) + 0.5*sum_i sum_j b_i b_j k(y_i, y_j)
    def kernel_loss(
            self, loss_component_expert, loss_component_training, a, b,
            dim_expert):
        K_expert = self.distances(
            loss_component_expert, loss_component_expert
        )  # (B,N,N)
        K_training = self.distances(
            loss_component_training, loss_component_training
        )  # (B,M,M)
        K_expert_training = self.distances(
            loss_component_expert, loss_component_training
        )  # (B,N,M)

        # (B,N)
        a_x = tf.squeeze(tf.matmul(K_expert, tf.expand_dims(a, axis=-1)))
        # (B,M)
        b_y = tf.squeeze(tf.matmul(K_training, tf.expand_dims(b, axis=-1)))
        # (B,N)
        b_x = tf.squeeze(tf.matmul(K_expert_training,
                                   tf.expand_dims(b, axis=-1)))
        A = 0.5 * self.scal(a, a_x)
        B = 0.5 * self.scal(b, b_y)
        C = self.scal(a, b_x)
        loss = A + B - C

        # average over batches
        mean_loss = tf.reduce_mean(loss)
        return mean_loss


MMD_energy = MmdEnergy()
