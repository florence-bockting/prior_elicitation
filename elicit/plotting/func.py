# SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# SPDX-License-Identifier: Apache-2.0

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf


def plot_loss(path):
    total_loss = pd.read_pickle(path+"/final_results.pkl")["loss"]
    loss_components = pd.read_pickle(
        path+"/final_results.pkl"
        )["loss_component"]
    if np.array(loss_components).shape[-1] == 1:
        plt.plot(total_loss)
        plt.xlabel("epochs")
        plt.ylabel("discrepancy loss")
        plt.title("Loss function")
        plt.show()
    else:
        _, axs = plt.subplots(1, 2, constrained_layout=True, sharex=True)
        axs[0].plot(total_loss)
        axs[1].plot(loss_components)
        axs[0].xlabel("epochs")
        axs[0].ylabel("discrepancy loss")
        plt.show()


def plot_convergence(path):
    hyp = pd.read_pickle(path+"/final_results.pkl")["hyperparameter"]
    hyp_val = tf.stack([hyp[k] for k in hyp], -1)
    hyp_key = [k for k in list(hyp.keys())]
    _, axs = plt.subplots(1, hyp_val.shape[-1], constrained_layout=True,
                          figsize=(6, 3), sharex=True)
    [axs[i].plot(hyp_val[:, i]) for i in range(hyp_val.shape[-1])]
    [axs[i].set_title(hyp_key[i]) for i in range(hyp_val.shape[-1])]
    [axs[i].set_xlabel("epochs") for i in range(hyp_val.shape[-1])]
    plt.show()
