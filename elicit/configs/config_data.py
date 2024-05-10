import polars as pl
import numpy as np
import tensorflow as tf
import pandas as pd
import patsy as pa
import numpy as np

from functions.helper_functions import save_as_pkl


def load_design_matrix_haberman(global_dict):
    scaling = global_dict["scaling"]
    select_obs = global_dict["select_obs"]
    # load dataset from repo
    from ucimlrepo import fetch_ucirepo 
    # fetch dataset 
    d_raw = fetch_ucirepo(id=43)["data"]
    # select predictors
    d_combi = d_raw["features"]
    # create new dataset with predictors and dependen variable
    d_combi["survival_status"] = d_raw["targets"]["survival_status"]
    # aggregate observations for Binomial format
    data = pl.DataFrame(d_combi).group_by("positive_auxillary_nodes").agg()
    # add intercept
    data_int = data.with_columns(intercept = pl.repeat(1, len(data["positive_auxillary_nodes"])))
    # reorder columns
    data_reordered = data_int.select(["intercept", "positive_auxillary_nodes"])
    # scale predictor if specified
    if scaling == "divide_by_std":
        sd = np.std(np.array(data_reordered["positive_auxillary_nodes"]))
        d_scaled = data_reordered.with_columns(X_scaled = np.array(data_reordered["positive_auxillary_nodes"])/sd)
        d_final = d_scaled.select(["intercept", "X_scaled"])
    if scaling == "standardize":
        sd = np.std(np.array(data_reordered["positive_auxillary_nodes"]))
        mean = np.mean(np.array(data_reordered["positive_auxillary_nodes"]))
        d_scaled = data_reordered.with_columns(X_scaled = (np.array(data_reordered["positive_auxillary_nodes"])-mean)/sd)
        d_final = d_scaled.select(["intercept", "X_scaled"])
    if scaling is None:
        d_final = data_reordered
    # select specific observations
    d_final = tf.gather(d_final, select_obs)
    # convert pandas data frame to tensor
    array = tf.cast(d_final, tf.float32)
    # save file in object
    path = global_dict["saving_path"]+'/design_matrix.pkl'
    save_as_pkl(array, path)
    return path

def load_design_matrix_equality(global_dict):
    # load dataset from repo
    url = "https://github.com/bayes-rules/bayesrules/blob/404fbdbae2957976820f9249e9cc663a72141463/data-raw/equality_index/equality_index.csv?raw=true"
    df = pd.read_csv(url)
    # exclude california from analysis as extreme outlier
    df_filtered = df.loc[df["state"] != "california"]
    # select predictors
    df_prep = df_filtered.loc[:, ["historical", "percent_urban"]]
    # get groups of cat. predictor
    groups = pd.DataFrame(np.asarray(pa.dmatrix("historical:percent_urban", df_prep)))
    # create design matrix of factor 
    design_matrix = pd.DataFrame(np.where(groups != 0, 1, 0), 
                                 columns = ["intercept","dem","gop","swing"])
    # use level=dem as baseline level
    design_matrix = design_matrix.loc[:,["intercept","gop","swing"]]
    # add continuous predictor to design matrix
    data_reordered = design_matrix.assign(percent_urban=df_prep.loc[:, "percent_urban"]
                                          ).sort_values(by=["gop","swing","percent_urban"]).dropna()
    # scale predictor if specified
    if global_dict["scaling"] == "divide_by_std":
        sd = np.std(np.array(data_reordered["percent_urban"]))
        d_scaled = data_reordered.assign(percent_urban_scaled = np.array(data_reordered["percent_urban"])/sd)
        d_final = d_scaled.loc[:,["intercept", "percent_urban_scaled","gop","swing"]]
    if global_dict["scaling"] == "standardize":
        sd = np.std(np.array(data_reordered["percent_urban"]))
        mean = np.mean(np.array(data_reordered["percent_urban"]))
        d_scaled = data_reordered.assign(percent_urban_scaled = (np.array(data_reordered["percent_urban"])-mean)/sd)
        d_final = d_scaled.loc[:, ["intercept", "percent_urban_scaled", "gop","swing"]]
    if global_dict["scaling"] is None:
        d_final = data_reordered
    # select specific observations
    d_final = tf.gather(d_final, global_dict["select_obs"])
    # convert pandas data frame to tensor
    array = tf.cast(d_final, tf.float32)
    # save file in object
    path = global_dict["saving_path"]+'/design_matrix.pkl'
    save_as_pkl(array, path)
    return path