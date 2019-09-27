import argparse
import multiprocessing
import os
import pickle
import time

# from .bayesopt import run_bayesopt
from .experiments_pbp import ExperimentPBPReg

# from dataclasses import dataclass


####################
## Set parameters ##
####################

dirname = os.path.dirname(__file__)
#################
## Define grid ##
#################

all_data_sets = (
    ["yacht" + str(i) for i in range(20)]
    + ["energy" + str(i) for i in range(20)]
    + ["boston" + str(i) for i in range(20)]
    + ["concrete" + str(i) for i in range(20)]
    + ["wine" + str(i) for i in range(20)]
    + ["kin8nm" + str(i) for i in range(20)]
    + ["naval" + str(i) for i in range(20)]
    + ["powerplant" + str(i) for i in range(20)]
)

large_data_sets = (
    ["wine" + str(i) for i in range(20)]
    + ["kin8nm" + str(i) for i in range(20)]
    + ["naval" + str(i) for i in range(20)]
    + ["powerplant" + str(i) for i in range(20)]
)

grid = [(data_set) for data_set in all_data_sets]


#######################
## Define BO process ##
#######################


# @dataclass
class multiproc_optim:
    def __init__(
        self,
        method=None,
        data_folder=None,
        results_folder=None,
        data_params=None,
        model_params=None,
        train_params=None,
        exp_prefix=None,
    ):
        self.method = method
        self.data_folder = data_folder
        self.results_folder = results_folder
        self.data_params = data_params
        self.model_params = model_params
        self.train_params = train_params
        self.exp_prefix = exp_prefix

    def __call__(self, data_set):
        start_clock_time = time.time()
        start_process_time = time.process_time()

        self.data_params["data_set"] = data_set

        experiment = ExperimentPBPReg(
            results_folder=self.results_folder,
            experiment_prefix=self.exp_prefix,
            data_folder=self.data_folder,
            data_set=self.data_params["data_set"],
            model_params=self.model_params,
            train_params=self.train_params,
            normalize_x=True,
            normalize_y=True,
        )
        experiment.run(log_metric_history=True)

        total_wall_clock_time = time.time() - start_clock_time
        total_process_time = time.process_time() - start_process_time
        experiment.save(
            save_final_metric=True,
            save_metric_history=True,
            # save_objective_history=True,
            save_model=True,
            # folder_path=folder,
        )

        with open(os.path.join(experiment.folder_name, "run_time.pkl"), "wb") as output:
            pickle.dump({'wall_clock': total_wall_clock_time,
                         'process_time': total_process_time},
                        output)

########################
## Run BO in parallel ##
########################
def main(args):

    # Data set
    data_params = {"data_set": None}

    # Model parameters
    model_params = {"hidden_sizes": args.hidden_sizes, "act_func": args.act}

    # Training parameters
    train_params = {"num_epochs": args.epochs, "seed": args.train_seed}

    multiproc_func = multiproc_optim(
        method=args.method,
        exp_prefix=args.prefix,
        data_folder=args.data_dir,
        results_folder=args.results,
        data_params=data_params,
        model_params=model_params,
        train_params=train_params,
    )

    start_time = time.time()

    # multiproc_func("naval10")
    mp = multiprocessing.get_context("forkserver")
    pool = mp.Pool(processes=args.jobs)
    pool.map(multiproc_func, grid)
    pool.close()
    pool.join()

    print("It took {:.2f}s".format(time.time() - start_time))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Bayesian NN for regression")

    parser.add_argument(
        "prefix",
        type=str,
        help="Prefix of experiment name in results folder Path: 'results/{prefix}_{method}/'",
    )

    parser.add_argument(
        "method",
        choices=["bbb", "pbp", "vadam", "dropout"],
        help="Inference method. Choices are: %(choices)s",
    )

    parser.add_argument(
        "--jobs",
        default=12,
        type=int,
        metavar="N",
        help="number of cross val experiments jobs to run (default: %(default)s)",
    )

    # Dataset
    parser.add_argument(  # data_folder
        "--data-dir",
        metavar="DIR",
        default=os.path.join(dirname, "../data/"),
        help="path to UCI datasets (default: %(default)s)",
    )
    parser.add_argument(  # data_params[n_splits]
        "--cv-splits",
        metavar="N",
        default=5,
        help="Number of cross validation splits (default: %(default)s)",
    )
    parser.add_argument(  # data_params[seed]
        "--cv-seed",
        metavar="N",
        default=123,
        help="RNG seed for cross validation splits (default: %(default)s)",
    )

    # Results
    parser.add_argument(  # results_folder
        "--results",
        metavar="DIR",
        type=str,
        default=os.path.join(dirname, "results/"),
        help="Path to folder storing results (default: %(default)s)",
    )

    # Loader
    # parser.add_argument( # train_params[batch_size]
    #     "-b",
    #     "--batch-size",
    #     default=32,
    #     type=int,
    #     metavar="N",
    #     help="Mini-batch size (default: %(default)s)",
    # )
    parser.add_argument(  # data_params[workers]
        "--data-workers",
        default=0,
        type=int,
        metavar="N",
        help="Number of data loading workers (default: %(default)s)",
    )

    # Training
    parser.add_argument(  # train_params[seed]
        "--train-seed",
        metavar="N",
        default=123,
        help="RNG seed for pytorch (default: %(default)s)",
    )
    parser.add_argument(  # train_params[num_epochs]
        "--epochs",
        metavar="N",
        type=int,
        default=40,
        help="Number of epochs to train for (default: %(default)s)",
    )
    parser.add_argument(  # train_params[evals_per_epoch]
        "--iters-epoch",
        metavar="N",
        type=int,
        default=1000,
        help="Number evals (iters) per epoch (default: %(default)s)",
    )
    parser.add_argument(  # train_params[eval_mc_samples]
        "--eval-samples",
        metavar="N",
        type=int,
        default=100,
        help="Number of MC samples during evaluation (default: %(default)s)",
    )
    # parser.add_argument(  # train_params[train_mc_samples]
    #     "train-samples",
    #     metavar="N",
    #     type=int,
    #     default=20,
    #     help="Number of MC samples during training (default: %(default)s)",
    # )

    # Optimizer parameters
    parser.add_argument(  # optim_params[learning_rate]
        "--lr",
        default=0.01,
        type=float,
        metavar="LR",
        help="Learning rate (default: %(default)s)",
    )

    parser.add_argument(  # optim_params[learning_rate]
        "--betas",
        default=(0.9, 0.99),
        type=tuple,
        metavar="BETA1 BETA2",
        help="Params of the optimizer (default: %(default)s)",
    )

    parser.add_argument(  # optim_params[prec_init]
        "--prec-init",
        default=10.0,
        type=float,
        metavar="PRECISION",
        help="Precision for variational distribution at init (default: %(default)s)",
    )

    # Model
    parser.add_argument(  # model_params[hidden_sizes]
        "--hidden-sizes",
        default=[50],
        type=list,
        metavar="[N1, N2, ...]",
        help="List w/ nb of hiddens units per layer (default: %(default)s)",
    )
    parser.add_argument(  # model_params[act_func]
        "--act",
        default="relu",
        type=str,
        help="Nonlinear activation function (default: %(default)s)",
    )

    parser.add_argument(
        "--bo-init",
        default=5,
        type=int,
        metavar="N",
        help="Number of steps of random exploration to perform (default: %(default)s)",
    )

    parser.add_argument(
        "--bo-iter",
        default=25,
        type=int,
        metavar="N",
        help="Number of steps of bayesian optimization to perform (default: %(default)s)",
    )

    args = parser.parse_args()
    print(args)
    main(args)
