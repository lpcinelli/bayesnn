import argparse
import os
import time
from dataclasses import dataclass

from torch import multiprocessing

from .bayesopt import run_bayesopt

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


@dataclass
class multiproc_optim:
    method: str
    data_folder: str
    results_folder: str
    data_params: dict
    model_params: dict
    train_params: dict
    optim_params: dict
    evals_per_epoch: int
    bo_params: dict
    length_scale: list
    nu: float
    alpha: float
    exp_prefix: str

    def __call__(self, data_set):
        self.data_params["data_set"] = data_set

        if data_set.startswith(("naval", "powerplant")):
            param_bounds = {"log_noise_prec": (1, 5), "log_prior_prec": (-3, 4)}
        else:
            param_bounds = {"log_noise_prec": (0, 5), "log_prior_prec": (-4, 4)}

        if self.method == "dropout":
            param_bounds["dropout"] = (0, 1)

        if data_set in large_data_sets:
            self.train_params["batch_size"] = 128
            self.train_params["train_mc_samples"] = 5 if self.method == "vadam" else 10
        else:
            self.train_params["batch_size"] = 32
            self.train_params["train_mc_samples"] = 10 if self.method == "vadam" else 20

        run_bayesopt(
            method=self.method,
            experiment_prefix=self.exp_prefix,
            data_folder=self.data_folder,
            results_folder=self.results_folder,
            data_params=self.data_params,
            model_params=self.model_params,
            train_params=self.train_params,
            optim_params=self.optim_params,
            evals_per_epoch=self.evals_per_epoch,
            param_bounds=param_bounds,
            bo_params=self.bo_params,
            length_scale=self.length_scale,
            nu=self.nu,
            alpha=self.alpha,
        )


########################
## Run BO in parallel ##
########################
def main(args):

    # Data set
    data_params = {
        "data_set": None, 
        "n_splits": args.cv_splits,
        "seed": args.cv_seed}
    
    # Model parameters
    model_params = {
        "hidden_sizes": args.hidden_sizes,
        "act_func": args.act,
        "prior_prec": None,
        "noise_prec": None,
    }
    
    # Training parameters
    train_params = {
        "num_epochs": args.epochs,
        "batch_size": None,
        "train_mc_samples": None,
        "eval_mc_samples": args.eval_samples,
        "seed": args.train_seed,
    }
    
    # Optimizer parameters
    optim_params = {
        "learning_rate": args.lr, 
        "betas": tuple(args.betas),
        "prec_init": args.prec_init
    }

    bo_params = {'acq': 'ei',
                 'init_points': args.bo_init, 
                 'n_iter': args.bo_iter}
    
    # Gaussian Process parameters
    length_scale = [1, 2]
    nu = 2.5
    alpha = 1e-2

    multiproc_func = multiproc_optim(
        method=args.method,
        exp_prefix=args.prefix,
        data_folder=args.data_dir,
        results_folder=args.results,
        data_params=data_params,
        model_params=model_params,
        train_params=train_params,
        optim_params=optim_params,
        evals_per_epoch=args.iters_epoch,
        bo_params=bo_params,
        length_scale=length_scale,
        nu=nu,
        alpha=alpha,
    )

    start_time = time.time()

    mp = multiprocessing.get_context("spawn")
    # mp = multiprocessing.get_context("forkserver")
    pool = mp.Pool(processes=args.jobs)
    pool.map(multiproc_func, ["yacht0", "yacht1", "yacht2"])
    pool.close()
    pool.join()

    print(f"It took {time.time() - start_time:.2f}s")


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
        help=f"Inference method. Choices are: %(choices)s",
    )


    parser.add_argument(
        "--jobs",
        default=12,
        type=int,
        metavar="N",
        help="number of cross val experiments jobs to run (default: %(default)s)",
    )

    # Dataset
    parser.add_argument( #data_folder
        "--data-dir",
        metavar="DIR",
        default=os.path.join(dirname, "../data/"),
        help="path to UCI datasets (default: %(default)s)",
    )
    parser.add_argument( # data_params[n_splits]
        "--cv-splits",
        metavar="N",
        default=5,
        help="Number of cross validation splits (default: %(default)s)",
    )
    parser.add_argument( # data_params[seed]
        "--cv-seed",
        metavar="N",
        default=123,
        help="RNG seed for cross validation splits (default: %(default)s)",
    )

    # Results
    parser.add_argument( # results_folder
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
    parser.add_argument( # data_params[workers]
        "--data-workers",
        default=0,
        type=int,
        metavar="N",
        help="Number of data loading workers (default: %(default)s)",
    )

    # Training
    parser.add_argument( # train_params[seed]
        "--train-seed",
        metavar="N",
        default=123,
        help="RNG seed for pytorch (default: %(default)s)",
    )
    parser.add_argument( # train_params[num_epochs]
        "--epochs",
        metavar="N",
        type=int,
        default=40,
        help="Number of epochs to train for (default: %(default)s)",
    )
    parser.add_argument( # train_params[evals_per_epoch]
        "--iters-epoch",
        metavar="N",
        type=int,
        default=1000,
        help="Number evals (iters) per epoch (default: %(default)s)",
    )
    parser.add_argument( # train_params[eval_mc_samples]
        "--eval-samples",
        metavar="N",
        type=int,
        default=100,
        help="Number of MC samples during evaluation (default: %(default)s)",
    )
    # parser.add_argument( # train_params[train_mc_samples]
    #     "train-samples",
    #     metavar="N",
    #     type=int,
    #     default=20,
    #     help="Number of MC samples during training (default: %(default)s)",
    # )

    # Optimizer parameters
    parser.add_argument( # optim_params[learning_rate]
        "--lr",
        default=0.01,
        type=float,
        metavar="LR",
        help="Learning rate (default: %(default)s)",
    )

    parser.add_argument( # optim_params[learning_rate]
        "--betas",
        default=(0.9, 0.99),
        type=tuple,
        metavar="BETA1 BETA2",
        help="Params of the optimizer (default: %(default)s)",
    )

    parser.add_argument( # optim_params[prec_init]
        "--prec-init",
        default=10.0,
        type=float,
        metavar="PRECISION",
        help="Precision for variational distribution at init (default: %(default)s)",
    )

    # Model
    parser.add_argument( # model_params[hidden_sizes]
        "--hidden-sizes",
        default=[50],
        type=list,
        metavar="[N1, N2, ...]",
        help="List w/ nb of hiddens units per layer (default: %(default)s)",
    )
    parser.add_argument(# model_params[act_func]
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
