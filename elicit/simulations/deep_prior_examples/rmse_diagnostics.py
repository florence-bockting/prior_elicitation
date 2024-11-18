import tensorflow as tf
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

scenario="independent"
path_sim = f"elicit/simulations/LiDO_cluster/sim_results/deep_prior/normal_{scenario}3"

all_files = os.listdir(path_sim)

def rmse(expert, model):
    return tf.sqrt(
        tf.reduce_mean(
            tf.square(tf.subtract(expert, model)),0
        ))

elicit_exp = pd.read_pickle(
    f"elicit/simulations/LiDO_cluster/experts/deep_{scenario}_normal/elicited_statistics.pkl"
    )


res_sim = dict()
for i in range(len(all_files)):
    for j in elicit_exp.keys():
        if j == "quantiles_logR2":
            elicit_e = tf.exp(pd.read_pickle(
                f"elicit/simulations/LiDO_cluster/experts/deep_{scenario}_normal/elicited_statistics.pkl"
                )[j])
        
            elicit = tf.stack(
                tf.exp(pd.read_pickle(path_sim+f"/{all_files[i]}"+"/elicited_statistics.pkl")[j])
                )
        else:
            elicit_e = pd.read_pickle(
                f"elicit/simulations/LiDO_cluster/experts/deep_{scenario}_normal/elicited_statistics.pkl"
                )[j]
            elicit = tf.stack(
                pd.read_pickle(path_sim+f"/{all_files[i]}"+"/elicited_statistics.pkl")[j]
                )
        res_sim[j+f".{i}"+".single"] = rmse(elicit_e, elicit)
        res_sim[j+f".{i}"+".mean"] = tf.reduce_mean(rmse(elicit_e, elicit))
    #res[f"sim_{i}"] = res_sim


res = dict(
    sim = [],
    target = [],
    stats = [],
    rmse = []
    )

for i in res_sim.keys():
    splitted = i.split(".")
    res["sim"].append(splitted[1])
    res["target"].append(splitted[0])
    res["stats"].append(splitted[2])
    res["rmse"].append(res_sim[i].numpy())

df = pd.DataFrame(res)


_, axs = plt.subplots(2,2, constrained_layout=True, figsize=(7,4),
                      sharex=True)
for i in range(len(all_files)):
    for r,c,j,l in zip([0,1,0,1], [0,0,1,1],
                     list(np.unique(res["target"]))[1:],
                     [r"$y_i \mid gr_1$", r"$y_i \mid gr_2$",r"$y_i \mid gr_3$",
                      r"$R^2$"]):
        
        df_mean = df[df["stats"] == "mean"]
        df_mean_sorted = df_mean[df_mean["target"] == j].sort_values(
            by="rmse", ascending=False)
        highest_rmse = df_mean_sorted.iloc[:3]["sim"]
        
        if str(i) in list(highest_rmse):
            idx = list(highest_rmse).index(str(i))
            label = f"{i}"
            shape = ["*", "^", "H"][idx]
            zorder=2
            col=["#c44601", "#00bf7d", "#8babf1"][idx]
        else:
            zorder=1
            label=""
            shape="."
            col="grey"
        df_value = df[(df["sim"]==str(i)) & (df["target"]==j)& (df["stats"]=="single")]
        df_mean = df[(df["sim"]==str(i)) & (df["target"]==j)& (df["stats"]=="mean")]
        
        axs[r,c].plot([0.05, 0.25, 0.5, 0.75, 0.95], list(df_value["rmse"])[0],
                 color=col,marker=shape,lw=0, zorder=zorder)
        axs[r,c].plot([1.2], list(df_mean["rmse"])[0], marker=shape, 
                      label=label, color=col, zorder=zorder, lw=0)
        axs[r,c].axvline(1.1, linestyle="solid", color=col, lw=1)
        axs[r,c].legend(title="seed:", frameon=False, ncol=3, 
                        columnspacing=0.3, fontsize="x-small", alignment="left")
        axs[r,c].set_xticks([0.05, 0.25, 0.5, 0.75, 0.95, 1.2], 
                       [".05", ".25", ".50", ".75", ".95", "mean"], fontsize="small")
        axs[r,c].spines[['right', 'top']].set_visible(False)
        axs[r,c].yaxis.set_tick_params(labelsize=7)
        axs[r,c].set_title(l)
[axs[i,j].set_ylim(0,2.) for i,j in zip([0,0,1],[0,1,0])]
[axs[i,0].set_ylabel("RMSE") for i in range(2)]
[axs[1,i].set_xlabel("quantiles") for i in range(2)]
