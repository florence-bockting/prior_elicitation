import polars as pl
import numpy as np
import tensorflow as tf
import pandas as pd
import patsy as pa


def load_design_matrix_haberman(scaling, selected_obs):
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
    # select only relevant observations
    if selected_obs is not None:
        d_final = tf.gather(d_final, selected_obs, axis = 0)
    # convert pandas data frame to tensor
    array = tf.cast(d_final, tf.float32)
    return array

def load_design_matrix_equality(scaling, selected_obs):
    # load dataset from repo
    url = "https://github.com/bayes-rules/bayesrules/blob/404fbdbae2957976820f9249e9cc663a72141463/data-raw/equality_index/equality_index.csv?raw=true"
    df = pd.read_csv(url)
    # exclude california from analysis as extreme outlier
    df_filtered = df.loc[df["state"] != "california"]
    # select predictors
    df_prep = df_filtered.loc[:, ["historical", "percent_urban"]]
    # reorder historical predictor
    df_reordered = df_prep.sort_values(["historical", "percent_urban"])
    # add dummy coded predictors
    df_reordered["gop"] = np.where(df_reordered["historical"] == "gop", 1, 0)
    df_reordered["swing"] = np.where(df_reordered["historical"] == "swing", 1, 0)
    df_reordered["intercept"] = 1
    # select only required columns
    data_reordered = df_reordered[["intercept", "percent_urban", "gop","swing"]]
    # scale predictor if specified
    if scaling == "divide_by_std":
        sd = np.std(np.array(data_reordered["percent_urban"]))
        d_scaled = data_reordered.assign(percent_urban_scaled = np.array(data_reordered["percent_urban"])/sd)
        d_final = d_scaled.loc[:,["intercept", "percent_urban_scaled","gop","swing"]]
    if scaling == "standardize":
        sd = np.std(np.array(data_reordered["percent_urban"]))
        mean = np.mean(np.array(data_reordered["percent_urban"]))
        d_scaled = data_reordered.assign(percent_urban_scaled = (np.array(data_reordered["percent_urban"])-mean)/sd)
        d_final = d_scaled.loc[:, ["intercept", "percent_urban_scaled", "gop","swing"]]
    if scaling is None:
        d_final = data_reordered
    if selected_obs is not None:
        # select only relevant observations
        d_final = tf.gather(d_final, selected_obs, axis = 0)
    # convert pandas data frame to tensor
    array = tf.cast(d_final, tf.float32)
    return array

def load_design_matrix_truth(n_group):
    # construct design matrix with 2-level and 3-level factor
    df =  pa.dmatrix("a*b", pa.balanced(a = 2, b = 3, repeat = n_group), 
                    return_type="dataframe")
    # save in correct format
    d_final = tf.cast(df, dtype = tf.float32)
    return d_final