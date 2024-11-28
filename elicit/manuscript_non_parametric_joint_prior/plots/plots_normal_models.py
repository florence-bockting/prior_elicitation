import pandas as pd

from elicit.plotting.sensitivity_func import plot_elicited_stats, plot_loss

# select scenario (one of: independent, correlated, skewed)
scenario = "correlated"

# common input arguments
seed = 1
# ranges for sensitivity analysis
mu0_seq = [0, 5, 10, 15, 20]
sigma0_seq = [0.1, 1.5, 2, 3.0, 4]
mu1_seq = [0, 5, 10, 15, 20]
sigma1_seq = [0.1, 1.5, 2, 3.0, 4]
mu2_seq = [0, 5, 10, 15, 20]
sigma2_seq = [0.1, 1.5, 2, 3.0, 4]
a_seq = [1, 5, 10, 20, 25]
b_seq = [1, 5, 10, 20, 25]

# ground truth hyperparameter values
input_dict = dict(
    mu0=10.0,
    sigma0=2.5,
    mu1=7.0,
    sigma1=1.3,
    mu2=2.5,
    sigma2=0.8,
    a=5.0,
    b=2.0,
    skew1=4.0,
    skew2=4.0,
    cor01=0.3,
    cor02=-0.3,
    cor12=-0.2,
)

# settings conditional on scenario:
if scenario == "independent":
    from elicit.manuscript_non_parametric_joint_prior.functions.independent_normal_prior_checks import (  # noqa
        prep_sensitivity_res,
        run_sensitivity,
        plot_sensitivity,
    )

    # optional parameters for easier plotting
    cor_seq = None
    skewness1_seq = None
    skewness2_seq = None
    # seed with highest slope for detailed inspection
    file_seed = "/normal_independent_28"
if scenario == "correlated":
    from elicit.manuscript_non_parametric_joint_prior.functions.correlated_normal_prior_checks import (  # noqa
        prep_sensitivity_res,
        run_sensitivity,
        plot_sensitivity,
    )

    # additional hyperparameter for sensitivity analysis
    cor_seq = [-0.8, -0.5, 0.0, 0.5, 0.8]
    skewness1_seq = None
    skewness2_seq = None
    # seed with highest slope for detailed inspection
    file_seed = "/normal_correlated_6"
if scenario == "skewed":
    from elicit.manuscript_non_parametric_joint_prior.functions.skewed_normal_prior_checks import (  # noqa
        prep_sensitivity_res,
        run_sensitivity,
        plot_sensitivity,
    )

    # additional hyperparameter for sensitivity analysis
    cor_seq = None
    skewness1_seq = [0.1, 2.0, 4.0, 8.0, 12.0]
    skewness2_seq = [0.1, 2.0, 4.0, 8.0, 12.0]
    # seed with highest slope for detailed inspection
    file_seed = "/normal_skewed_27"

from elicit.manuscript_non_parametric_joint_prior.functions.convergence_diagnostics import (  # noqa
    plot_conv_diagnostics,
)
from elicit.manuscript_non_parametric_joint_prior.functions.preprocess_sim_res_norm import (  # noqa
    prep_sim_res,
)
from elicit.manuscript_non_parametric_joint_prior.functions.scenarios_normal_model_averaging import (  # noqa
    run_model_averaging,
    plot_learned_priors,
)


# path to oracle data and to simulation results
# data can be found in OSF https://osf.io/xrzh6/
# download the zip, unpack folder locally and adjust the
# path variable accordingly
path_sim_res = f"elicit/manuscript_non_parametric_joint_prior/simulation_results/normal_{scenario}"  # noqa
path_expert = f"elicit/manuscript_non_parametric_joint_prior/experts/deep_{scenario}_normal"  # noqa
path_sensitivity_res = f"manuscript_non_parametric_joint_prior/sensitivity_results/normal_{scenario}_sensitivity"  # noqa

# read oracle data
prior_expert = pd.read_pickle(path_expert + "/prior_samples.pkl")


# %% SENSITIVITY ANALYSIS
# run sensitivity analysis
# Note: data used in manuscript are provided in OSF (see file header)

# run sensitivity analysis
run_sensitivity(
    seed,
    path_sensitivity_res,
    mu0_seq,
    mu1_seq,
    mu2_seq,
    sigma0_seq,
    sigma1_seq,
    sigma2_seq,
    a_seq,
    b_seq,
    cor_seq,
    skewness1_seq,
    skewness2_seq,
)

# if you  want to use data from manuscript:
df_sim_res = prep_sensitivity_res(path_sensitivity_res, input_dict)

# plot results
plot_sensitivity(
    df_sim_res,
    mu0_seq,
    mu1_seq,
    mu2_seq,
    sigma0_seq,
    sigma1_seq,
    sigma2_seq,
    a_seq,
    b_seq,
    cor_seq,
    skewness1_seq,
    skewness2_seq,
)


# %% CONVERGENCE ANALYSIS
# plot slopes per seed
plot_conv_diagnostics(
    path_sim_res, start=1400, end=1500, scenario=scenario, last_vals=200
)

# preprocessing of simulation results
(
    cor_res_agg,
    prior_res_agg,
    elicits_gr1_agg,
    elicits_gr2_agg,
    elicits_gr3_agg,
    elicits_r2_agg,
) = prep_sim_res(path_sim_res, scenario)

# plot elicited statistics
plot_elicited_stats(
    prior_expert,
    path_expert,
    prior_res_agg,
    elicits_gr1_agg,
    elicits_gr2_agg,
    elicits_gr3_agg,
    elicits_r2_agg,
    cor_res_agg,
    scenario,
    save_fig=False,
)

# plot convergence of all quantities of interest and learned prior for seed
# with highest slope
plot_loss(path_sim_res, path_expert, file_seed, scenario, save_fig=False)


# %% SIMULATION RESULTS / LEARNED PRIORS
# run model averaging method
(w_MMD, prior_samples, averaged_priors, min_weight, max_weight) = run_model_averaging( # noqa
    path_sim_res, B=128, sim_from_prior=200, num_param=4
)  # noqa
# plot learned marginal priors per seed and average prior
plot_learned_priors(
    scenario,
    prior_expert,
    w_MMD,
    prior_samples,
    averaged_priors,
    min_weight,
    max_weight,
)
