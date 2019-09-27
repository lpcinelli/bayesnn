import os
import pickle

import numpy as np
from scipy import stats

##################
## Define funcs ##
##################

def folder_name(experiment_name, param_bounds, bo_params, data_params, model_params, train_params, optim_params, results_folder="./results"):
    pp = ''.join('{}:{}|'.format(key, val) for key, val in sorted(param_bounds.items()))[:-1]
    bp = ''.join('{}:{}|'.format(key, val) for key, val in sorted(bo_params.items()))[:-1]
    dp = ''.join('{}:{}|'.format(key, val) for key, val in sorted(data_params.items()))[:-1]
    mp = ''.join('{}:{}|'.format(key, val) for key, val in sorted(model_params.items()))[:-1]
    tp = ''.join('{}:{}|'.format(key, val) for key, val in sorted(train_params.items()))[:-1]
    op = ''.join('{}:{}|'.format(key, val) for key, val in sorted(optim_params.items()))[:-1]
    return os.path.join(results_folder, experiment_name, pp, bp, dp, mp, tp, op)


#####################
## Define datasets ##
#####################

ds = ("boston", "concrete", "energy", "kin8nm", "naval", "powerplant", "wine", "yacht")

for data_sets in ds:

    # Folder containing results
    results_folder = "./results/"

    # Data set
    data_params = {'data_set': None,
                   'n_splits': 5,
                   'seed': 123}


    # Model parameters
    model_params = {'hidden_sizes': [50],
                    'act_func': 'relu',
                    'prior_prec': None,
                    'noise_prec': None}

    # BO parameters
    bo_params = {'acq': 'ei',
                 'init_points': 5,
                 'n_iter': 25}

    if data_sets in ("naval", "powerplant"):
        param_bounds = {'log_noise_prec': (1, 5),
                        'log_prior_prec': (-3, 4)}
    else:
        param_bounds = {'log_noise_prec': (0, 5),
                        'log_prior_prec': (-4, 4)}

    if data_sets in ("kin8nm", "naval", "powerplant", "wine"):
        # batch size
        bbb_batch = 128
        bbb_mc_eval = 10

        #
        vadam_batch = 128
        vadam_mc_eval = 5
    else:
        bbb_batch = 32
        bbb_mc_eval = 20

        vadam_batch = 32
        vadam_mc_eval = 10

    #######################
    ## Load BBB  results ##
    #######################

    experiment_name = "bayesopt_bbb"

    # Training parameters
    train_params = {'num_epochs': 40,
                    'batch_size': bbb_batch,
                    'train_mc_samples': bbb_mc_eval,
                    'eval_mc_samples': 100,
                    'seed': 123}

    # Optimizer parameters
    optim_params = {'learning_rate': 0.01,
                    'betas': (0.9,0.99),
                    'prec_init': 10.0}

    grid_marginalize = [(data_set) for data_set in [data_sets + str(i) for i in range(20)]]
    bbm_loglike = np.zeros([len(grid_marginalize)])
    bbm_rmse = np.zeros([len(grid_marginalize)])

    for i, (data_set) in enumerate(grid_marginalize):

        data_params['data_set'] = data_set
        folder = folder_name(results_folder = results_folder,
                             experiment_name = experiment_name,
                             param_bounds = param_bounds,
                             bo_params = bo_params,
                             data_params = data_params,
                             model_params = model_params,
                             train_params = train_params,
                             optim_params = optim_params)

        pkl_file = open(os.path.join(folder, 'final_metric.pkl'), 'rb')
        final_metric = pickle.load(pkl_file)
        pkl_file.close()

        bbm_loglike[i] = final_metric['test_pred_logloss'][-1]
        bbm_rmse[i] = final_metric['test_pred_rmse'][-1]

    ########################
    ## Load Vadam results ##
    ########################

    experiment_name = "bayesopt_vadam"

    # Training parameters
    train_params = {'num_epochs': 40,
                    'batch_size': vadam_batch,
                    'train_mc_samples': vadam_mc_eval,
                    'eval_mc_samples': 100,
                    'seed': 123}

    # Optimizer parameters
    optim_params = {'learning_rate': 0.01,
                    'betas': (0.9,0.99),
                    'prec_init': 10.0}

    grid_marginalize = [(data_set) for data_set in [data_sets + str(i) for i in range(20)]]
    vadam_loglike = np.zeros([len(grid_marginalize)])
    vadam_rmse = np.zeros([len(grid_marginalize)])

    for i, (data_set) in enumerate(grid_marginalize):

        data_params['data_set'] = data_set
        folder = folder_name(results_folder = results_folder,
                             experiment_name = experiment_name,
                             param_bounds = param_bounds,
                             bo_params = bo_params,
                             data_params = data_params,
                             model_params = model_params,
                             train_params = train_params,
                             optim_params = optim_params)

        pkl_file = open(os.path.join(folder, 'final_metric.pkl'), 'rb')
        final_metric = pickle.load(pkl_file)
        pkl_file.close()

        vadam_loglike[i] = final_metric['test_pred_logloss'][-1]
        vadam_rmse[i] = final_metric['test_pred_rmse'][-1]



    #############################
    ## Load MC Dropout results ##
    #############################

    experiment_name = "bayesopt_bbb"

    # Training parameters
    train_params = {'num_epochs': 40,
                    'batch_size': bbb_batch,
                    'train_mc_samples': bbb_mc_eval,
                    'eval_mc_samples': 100,
                    'seed': 123}

    # Optimizer parameters
    optim_params = {'learning_rate': 0.01,
                    'betas': (0.9,0.99),
                    'prec_init': 10.0}

    grid_marginalize = [(data_set) for data_set in [data_sets + str(i) for i in range(20)]]
    mcdrop_loglike = np.zeros([len(grid_marginalize)])
    mcdrop_rmse = np.zeros([len(grid_marginalize)])

    for i, (data_set) in enumerate(grid_marginalize):

        data_params['data_set'] = data_set
        folder = folder_name(results_folder = results_folder,
                             experiment_name = experiment_name,
                             param_bounds = param_bounds,
                             bo_params = bo_params,
                             data_params = data_params,
                             model_params = model_params,
                             train_params = train_params,
                             optim_params = optim_params)

        pkl_file = open(os.path.join(folder, 'final_metric.pkl'), 'rb')
        final_metric = pickle.load(pkl_file)
        pkl_file.close()

        mcdrop_loglike[i] = final_metric['test_pred_logloss'][-1]
        mcdrop_rmse[i] = final_metric['test_pred_rmse'][-1]

    ############################
    ## Prepare means and stds ##
    ############################

    mcdrop_loglikem = -np.mean(mcdrop_loglike, axis=0)
    mcdrop_loglikes = np.std(mcdrop_loglike, axis=0)/np.sqrt(20)
    bbm_rmsem = np.mean(bbm_rmse, axis=0)
    bbm_rmses = np.std(bbm_rmse, axis=0)/np.sqrt(20)

    vadam_loglikem = -np.mean(vadam_loglike, axis=0)
    vadam_loglikes = np.std(vadam_loglike, axis=0)/np.sqrt(20)
    vadam_rmsem = np.mean(vadam_rmse, axis=0)
    vadam_rmses = np.std(vadam_rmse, axis=0)/np.sqrt(20)

    pval_test = 0.01
    n_decimals = 2
    fmt_str = '{:0.' + str(n_decimals) + 'f}'

    ##############################
    ## t-test for log-liklihood ##
    ##############################

    if vadam_loglikem > bbm_loglikem:
        # compare to vadam
        _, p_value_b = stats.ttest_rel(vadam_loglike, bbm_loglike)

        sig_b = p_value_b <= pval_test

        print("\n\n\n", data_sets, ", Vadam is best with ll = ", fmt_str.format(vadam_loglikem), "$\pm$", fmt_str.format(vadam_loglikes))
        if sig_b:
            print("BBVI is significantly worse with ll = ", fmt_str.format(bbm_loglikem), "$\pm$", fmt_str.format(bbm_loglikes), ", pval = ", p_value_b)
        else:
            print("BBVI is comparable with ll = ", fmt_str.format(bbm_loglikem), "$\pm$", fmt_str.format(bbm_loglikes), ", pval = ", p_value_b)

    elif bbm_loglikem > vadam_loglikem:
        # compare to bbvi
        _, p_value_v = stats.ttest_rel(bbm_loglike, vadam_loglike)

        sig_v = p_value_v <= pval_test

        print("\n\n\n", data_sets, ", BBVI is best with ll = ", fmt_str.format(bbm_loglikem), "$\pm$", fmt_str.format(bbm_loglikes))
        if sig_v:
            print("Vadam is significantly worse with ll = ", fmt_str.format(vadam_loglikem), "$\pm$", fmt_str.format(vadam_loglikes), ", pval = ", p_value_v)
        else:
            print("Vadam is comparable with ll = ", fmt_str.format(vadam_loglikem), "$\pm$", fmt_str.format(vadam_loglikes), ", pval = ", p_value_v)

    else:
        print("error!")

    #####################
    ## t-test for rmse ##
    #####################

    if vadam_rmsem < bbm_rmsem :
        # compare to vadam
        _, p_value_b = stats.ttest_rel(vadam_rmse, bbm_rmse)

        sig_b = p_value_b <= pval_test

        print("\n", data_sets, ", Vadam is best with rmse = ", fmt_str.format(vadam_rmsem), "$\pm$", fmt_str.format(vadam_rmses))
        if sig_b:
            print("BBVI is significantly worse with rmse = ", fmt_str.format(bbm_rmsem), "$\pm$", fmt_str.format(bbm_rmses), ", pval = ", p_value_b)
        else:
            print("BBVI is comparable with rmse = ", fmt_str.format(bbm_rmsem), "$\pm$", fmt_str.format(bbm_rmses), ", pval = ", p_value_b)

    elif bbm_rmsem < vadam_rmsem:
        # compare to bbvi
        _, p_value_v = stats.ttest_rel(bbm_rmse, vadam_rmse)

        sig_v = p_value_v <= pval_test

        print("\n", data_sets, ", BBVI is best with rmse = ", fmt_str.format(bbm_rmsem), "$\pm$", fmt_str.format(bbm_rmses))
        if sig_v:
            print("Vadam is significantly worse with rmse = ", fmt_str.format(vadam_rmsem), "$\pm$", fmt_str.format(vadam_rmses), ", pval = ", p_value_v)
        else:
            print("Vadam is comparable with rmse = ", fmt_str.format(vadam_rmsem), "$\pm$", fmt_str.format(vadam_rmses), ", pval = ", p_value_v)

    else:
        print("error!")
