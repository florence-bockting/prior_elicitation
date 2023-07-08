
from plot_results import plot_results_lm, plot_results_binomial, plot_results_negbinom, plot_identifiability, plot_results_poisson
from loss_components import extract_loss_components
from losses import energy_loss
from trainer import trainer
from trainer_step import trainer_step
from configuration_global import settings

# binom, linreg, pois, negbinom, mlm, weib
selected_model = "negbinom"

# import settings
(prob_model, plot_diag, input_settings_global, input_settings_learning, 
 input_settings_loss, input_settings_model) = settings(selected_model)

# initialize the probabilistic model
generative_model = prob_model(input_settings_model, input_settings_learning,
                        input_settings_global)

# sample data from data generating model representing the expert input
target_quant_exp = generative_model.data_generating_model(
                        mus = input_settings_model["hyperparameter"]["mus"],
                        sigmas = input_settings_model["hyperparameter"]["sigmas"],
                        sigma_taus = input_settings_model["hyperparameter"]["sigma_taus"],
                        alpha_LKJ = input_settings_model["hyperparameter"]["alpha_LKJ"],
                        lambda0 = input_settings_model["hyperparameter"]["lambda0"],
                        input_settings_global = input_settings_global,
                        input_settings_learning = input_settings_learning,
                        input_settings_model = input_settings_model,
                        model_type = "expert")

if selected_model == "mlm":
    input_settings_model["model_specific"]["R2_0"] = target_quant_exp["R2_0"]
    input_settings_model["model_specific"]["R2_1"] = target_quant_exp["R2_1"]

# initialize general optimization workflow (expert / simulation model)
training = trainer(selected_model, generative_model, target_quant_exp, 
                   trainer_step, energy_loss, extract_loss_components)

# run optimization algorithm
(out, var, target_quant_sim, elicited_quant_sim,  elicited_quant_sim_ini, elicited_quant_exp, weights, 
 time_per_epoch, final_time) = training(generative_model, 
                                        input_settings_model, 
                                        input_settings_learning, 
                                        input_settings_global, 
                                        input_settings_loss)

# save final simulations
import pickle

obj_list = [out, var, target_quant_sim, elicited_quant_sim,  elicited_quant_sim_ini, elicited_quant_exp, target_quant_exp,
            weights,  time_per_epoch, final_time, final_time, input_settings_model,  input_settings_learning, 
            input_settings_global,  input_settings_loss]

file_name = f"../simulations/sim_{selected_model}.pkl"

open_file = open(file_name, "wb")
pickle.dump(obj_list, open_file)
open_file.close()

# read saved file
# open_file = open(file_name, "rb")
# loaded_list = pickle.load(open_file)
# open_file.close()
# print(loaded_list)

                                                                      
# plot diagnostics
plot_diag(var, out, input_settings_learning["epochs"])
plot_identifiability(var, vals=100)

# plot results
if selected_model == "linreg":
    samples_d = target_quant_sim
    xrange0=[0.0, 5.0]
    xrange1=[-0.4, 0.3]
    fct_b_lvl=3
    fct_a_lvl=2
    
    plot_results_lm(input_settings_model, input_settings_learning, var, 
                    input_settings_global, samples_d, xrange0, xrange1, 
                    fct_b_lvl, fct_a_lvl)

if selected_model == "binom":
    xrge1 = [-1.,1.]
    ylim = [7,22]
    model_samples = elicited_quant_sim
    expert_samples = elicited_quant_exp
    loss_format = "quantiles"
    
    plot_results_binomial(input_settings_model, input_settings_learning, 
                          input_settings_global, var, expert_samples, 
                          model_samples, xrge1, ylim, loss_format)    

if selected_model == "pois":
    expert_samples = elicited_quant_exp
    ylim = [0,35]
    
    plot_results_poisson(input_settings_model, input_settings_learning, 
                         input_settings_global, var, expert_samples, ylim)
    
if selected_model == "negbinom":
    xrg = [-3., 3.7]
    xrg1 = [0., 50.]
    expert_samples = elicited_quant_exp
    
    plot_results_negbinom(input_settings_model, input_settings_learning,
                          input_settings_global, var, expert_samples,
                          xrg,xrg1)


plot_res = plot_results_binomial

plot_res(mus = input_settings_model["hyperparameter"]["mus"], 
          sigmas = input_settings_model["hyperparameter"]["sigmas"],
          # lambda0 = input_settings_model["hyperparameter"]["lambda0"],
          var = var,
          X = input_settings_model["X"],
          model_samples = elicited_quant_sim,
          expert_samples = elicited_quant_exp,#["y_idx_loss"],
          epochs = input_settings_learning["epochs"],
          l_values=input_settings_global["l_values"],
          loss_format = "quantiles",
          # lm
         #  xrange0=[0.0, 5.0],
         #  xrange1=[-0.4, 0.3],
         #  fct_b_lvl=3,
         #  fct_a_lvl=2,
         #  samples_d = raw_samples_sim,
          # binom
          xrge1 = [-1.,1.],
          ylim = [7,22]
          # pois
        #  ylim = [0,35],
          # negbinom
    #     xrg=[-3., 3.7],
    #     xrg1=[0., 50.]
    )
    
# var[0][1][-1]
# tf.exp(var[1][1][-1])
         
# if select_model == 0:
#     # binomial model
#     plot_results_binomial(mus = input_settings_model["hyperparameter"]["mus"], 
#                           sigmas = input_settings_model["hyperparameter"]["sigmas"], 
#                           var = var, 
#                           X_cont = input_settings_model["X"], 
#                           expert_samples = samples, 
#                           epochs=input_settings_learning["epochs"], 
#                           idx=input_settings_model["X_idx"], 
#                           l_values=input_settings_global["l_values"],
#                           xrge1 = [-1.,1.], 
#                           ylim=[0,20])
# if select_model == 1:
#     # linear model
# plot_results_lm(var, raw_samples_sim, input_settings_learning["epochs"], 
#             lambda0=input_settings_model["hyperparameter"]["lambda0"], 
#                 mus=input_settings_model["hyperparameter"]["mus"], 
#                 sigmas=input_settings_model["hyperparameter"]["sigmas"],
#                       input_settings_global["l_values"], 
#                       xrange0=[0.0, 5.0],xrange1=[-0.4, 0.3],
#                       fct_b_lvl=3,fct_a_lvl=2)
# if select_model == 2:
#     # poisson model    
#     plot_results_poisson(var, samples, input_settings_model["X"], input_settings_learning["epochs"],
#                          input_settings_model["hyperparameter"]["mus"], 
#                          input_settings_model["hyperparameter"]["sigmas"],
#                          input_settings_model["X_idx"], input_settings_global["l_values"], 
#                          ylim = [0,35])

# if select_model == 3:
#     # negative binomial model
#     plot_results_negbinom(var, samples, input_settings_model["X"], 
#                           input_settings_learning["epochs"],
#                           input_settings_model["hyperparameter"]["mus"], 
#                           input_settings_model["hyperparameter"]["sigmas"],
#                           input_settings_model["hyperparameter"]["lambda0"], 
#                           input_settings_model["X_idx"], input_settings_global["l_values"], 
#                           xrg=[-3., 3.7],xrg1=[0., 50.])



