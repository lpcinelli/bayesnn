import copy
import math
import os
import pickle
import time

from .experiments import ExperimentBBBMLPReg

dirname = os.path.dirname(__file__)
# Folder for storing results
# results_folder = "./results/"
results_folder = os.path.join(dirname, 'results/')

# Folder containing data
# data_folder = "./../data"
data_folder = os.path.join(dirname, '../data/')

# Data set
data_params = {'data_set': 'yacht0',
               'n_splits': 5,
               'seed': 123}

# Model parameters
model_params = {'hidden_sizes': [50],
                'act_func': 'relu',
                'prior_prec': math.exp(-4.0), # precision of prior dist. p(w)
                'noise_prec': math.exp(3.3718961338247531)} # observation noise

# Training parameters
train_params = {'num_epochs': 40,
                'batch_size': 32,
                'train_mc_samples': 20,
                'eval_mc_samples': 100,
                'seed': 123}

# Optimizer parameters
optim_params = {'learning_rate': 0.01,
                'betas': (0.9, 0.99),
                'prec_init': 10.0} # for BBB experiment: init value for precision of posterior dist. q(w)

# Evaluations per epoch
evals_per_epoch = 1000



experiment = ExperimentBBBMLPReg(results_folder = results_folder,
                                data_folder = data_folder,
                                data_set = data_params['data_set'],
                                model_params = model_params,
                                train_params = train_params,
                                optim_params = optim_params,
                                evals_per_epoch = evals_per_epoch,
                                normalize_x = True,
                                normalize_y = True)

start_time = time.time()
experiment.run(log_metric_history = True)
elapsed_time = time.time() - start_time

print(f'It took {elapsed_time:.4f}s to run {data_params["data_set"]} for {train_params["num_epochs"]} epochs')

experiment.save(save_final_metric = True,
                save_metric_history = True,
                save_objective_history = True,
                save_model = False,
                save_optimizer = False)
