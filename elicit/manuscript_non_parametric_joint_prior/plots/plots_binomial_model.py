import pandas as pd


from elicit.manuscript_non_parametric_joint_prior.functions.independent_binomial_prior_checks import ( # noqa
    run_sensitivity_binom,
    prep_sensitivity_res,
    plot_sensitivity_binom,
)
from elicit.manuscript_non_parametric_joint_prior.functions.convergence_diagnostics import ( # noqa
    plot_conv_diagnostics,
)
from elicit.manuscript_non_parametric_joint_prior.functions.preprocess_sim_res_binom import ( # noqa
    prep_sim_res_binom,
)
from elicit.manuscript_non_parametric_joint_prior.functions.binomial_model_averaging import ( # noqa
    run_model_averaging,
    plot_learned_priors,
)
from elicit.plotting.sensitivity_func import (
    plot_elicited_stats_binom,
    plot_loss_binom,
    plot_prior_binom,
)


# path to oracle data and to simulation results
# data can be found in OSF https://osf.io/xrzh6/
# download the zip, unpack folder locally and adjust the
# path variable accordingly
path_sim_res = (
    "elicit/manuscript_non_parametric_joint_prior/simulation_results/binomial"
)
path_expert = "elicit/manuscript_non_parametric_joint_prior/experts/deep_binomial" # noqa
path_sensitivity_res = "elicit/manuscript_non_parametric_joint_prior/sensitivity_results/binomial_independent_sensitivity" # noqa

# read oracle data
prior_expert = pd.read_pickle(path_expert + "/prior_samples.pkl")


# %% SENSITIVITY ANALYSIS
# input arguments
seed = 1
mu0_seq = [-0.4, -0.2, 0.0, 0.2, 0.4]
mu1_seq = [-0.4, -0.2, 0.0, 0.2, 0.4]
sigma0_seq = [0.01, 0.1, 0.3, 0.6, 1.0]
sigma1_seq = [0.01, 0.1, 0.3, 0.6, 1.0]

# run sensitivity analysis
run_sensitivity_binom(
    seed, path_sensitivity_res, mu0_seq, mu1_seq, sigma0_seq, sigma1_seq
)

# Note: data used in manuscript are provided in OSF (see file header)
# if you  want to use data from manuscript:
df_sim_res = prep_sensitivity_res(path_sensitivity_res)

# plot results
plot_sensitivity_binom(df_sim_res)


# %% CONVERGENCE ANALYSIS
# plot slopes per seed
plot_conv_diagnostics(path_sim_res, start=500, end=600, last_vals=200)

# preprocessing of simulation results
cor_res_agg, prior_res_agg, elicit_res_agg, mean_res_agg, sd_res_agg = (
    prep_sim_res_binom(path_sim_res)
)

# plot elicited statistics
plot_elicited_stats_binom(
    prior_expert,
    path_expert,
    path_sim_res,
    elicit_res_agg,
    prior_res_agg,
    cor_res_agg,
    save_fig=False,
)
# plot convergence of total loss and loss components for seed with
# highest slope
plot_loss_binom(path_sim_res, path_expert, "/binomial_11", save_fig=False)
# plot convergence of all quantities of interest and learned prior for seed
# with highest slope
plot_prior_binom(path_sim_res, path_expert, "/binomial_11", save_fig=False)


# %% SIMULATION RESULTS / LEARNED PRIORS
# run model averaging method
w_MMD, prior_samples, averaged_priors, min_weight = run_model_averaging(
    path_sim_res, B=128, sim_from_prior=200, num_param=2
)
# plot learned marginal priors per seed and average prior
plot_learned_priors(w_MMD, prior_samples, averaged_priors, min_weight,
                    prior_expert)
