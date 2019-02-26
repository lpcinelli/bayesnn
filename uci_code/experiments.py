import os
import pickle

import numpy as np
import torch
from torch.optim import Adam

from .. import metrics
from ..callbacks import Timemeasure
from ..datasets import DEFAULT_DATA_FOLDER, Dataset
from ..models import BNN, MLP, DropoutMLP
from ..vadam.optimizers import Vadam, Vprop

###############################################################################
## Define function that specify folder naming convention for storing results ##
###############################################################################


def folder_name(
    experiment_name,
    data_set,
    model_params,
    train_params,
    optim_params,
    results_folder="./results",
):
    mp = "".join(
        "{}:{}|".format(key, val) for key, val in sorted(model_params.items())
    )[:-1]
    tp = "".join(
        "{}:{}|".format(key, val) for key, val in sorted(train_params.items())
    )[:-1]
    op = "".join(
        "{}:{}|".format(key, val) for key, val in sorted(optim_params.items())
    )[:-1]
    return os.path.join(results_folder, experiment_name, data_set, mp, tp, op)


###############################################################
## Define function for loading only history from experiments ##
###############################################################


def load_final_metric(
    experiment_name,
    data_set,
    model_params,
    train_params,
    optim_params,
    results_folder="./results",
    silent_fail=False,
):

    # Folder to load history from
    folder = folder_name(
        experiment_name,
        data_set,
        model_params,
        train_params,
        optim_params,
        results_folder,
    )
    file = os.path.join(folder, "final_metric.pkl")
    # Silent fail
    if silent_fail and not os.path.exists(file):
        return None

    # Load history
    pkl_file = open(file, "rb")
    final_metric = pickle.load(pkl_file)
    pkl_file.close()

    # Return history
    return final_metric


def load_metric_history(
    experiment_name,
    data_set,
    model_params,
    train_params,
    optim_params,
    results_folder="./results",
    silent_fail=False,
):

    # Folder to load history from
    folder = folder_name(
        experiment_name,
        data_set,
        model_params,
        train_params,
        optim_params,
        results_folder,
    )
    file = os.path.join(folder, "metric_history.pkl")

    # Silent fail
    if silent_fail and not os.path.exists(file):
        return None

    # Load history
    pkl_file = open(file, "rb")
    metric_history = pickle.load(pkl_file)
    pkl_file.close()

    # Return history
    return metric_history


def load_objective_history(
    experiment_name,
    data_set,
    model_params,
    train_params,
    optim_params,
    results_folder="./results",
    silent_fail=False,
):

    # Folder to load history from
    folder = folder_name(
        results_folder,
        experiment_name,
        data_set,
        model_params,
        train_params,
        optim_params,
    )
    file = os.path.join(folder, "objective_history.pkl")

    # Silent fail
    if silent_fail and not os.path.exists(file):
        return None

    # Load history
    pkl_file = open(file, "rb")
    objective_history = pickle.load(pkl_file)
    pkl_file.close()

    # Return history
    return objective_history


######################################
## Define abstract experiment class ##
######################################


class Experiment:
    def __init__(
        self,
        data_set,
        model_params,
        train_params,
        optim_params,
        evals_per_epoch=1,
        normalize_x=False,
        normalize_y=False,
        results_folder="./results",
        data_folder=DEFAULT_DATA_FOLDER,
        use_cuda=torch.cuda.is_available(),
        print_freq=5,
    ):

        # Store parameters
        self.data_set = data_set
        self.model_params = model_params
        self.train_params = train_params
        self.optim_params = optim_params
        self.evals_per_epoch = evals_per_epoch
        self.normalize_x = normalize_x
        self.normalize_y = normalize_y
        self.data_folder = data_folder
        self.results_folder = results_folder
        self.use_cuda = use_cuda
        self.print_freq = print_freq

        # Set random seed
        seed = train_params["seed"]
        torch.manual_seed(seed)
        if self.use_cuda:
            torch.cuda.manual_seed_all(seed)

        # Initialize metric history
        self.objective_history = []

        # Initialize data
        self.data = Dataset(data_set=data_set, data_folder=data_folder)

        ## All subclasses should override:

        # Define folder name for results
        self.folder_name = None

        # Define prediction function
        self.prediction = None

        # Define objective
        self.objective = None

        # Initialize model
        self.model = None

        # Initialize optimizer
        self.optimizer = None

        # Initialize metric history
        self.metric_history = None

        # Initialize final metric
        self.final_metric = None

    def run(self, log_metric_history=True):

        # Prepare
        callback = Timemeasure(self.data_set, self.print_freq)
        num_epochs = self.train_params["num_epochs"]
        batch_size = self.train_params["batch_size"]
        seed = self.train_params["seed"]
        callback.on_begin()

        # Set random seed
        torch.manual_seed(seed)
        if self.use_cuda:
            torch.cuda.manual_seed_all(seed)

        # Prepare data loader for training
        train_loader = self.data.get_train_loader(batch_size=batch_size)

        # Load full data set for evaluation
        x_train, y_train = self.data.load_full_train_set(use_cuda=self.use_cuda)
        x_test, y_test = self.data.load_full_test_set(use_cuda=self.use_cuda)

        # Compute normalization of x
        if self.normalize_x:
            self.x_means = torch.mean(x_train, dim=0)
            self.x_stds = torch.std(x_train, dim=0)
            self.x_stds[self.x_stds == 0] = 1

        # Compute normalization of y
        if self.normalize_y:
            self.y_mean = torch.mean(y_train)
            self.y_std = torch.std(y_train)
            if self.y_std == 0:
                self.y_std = 1

        # Set iterations for evaluation
        num_batches = np.ceil(self.data.get_train_size() / batch_size)
        if self.evals_per_epoch > num_batches:
            evals_per_epoch = num_batches
        else:
            evals_per_epoch = self.evals_per_epoch
        eval_iters = np.round(
            (1 + np.arange(evals_per_epoch)) * (num_batches / evals_per_epoch)
        ).astype(int)

        # Train model
        for epoch in range(num_epochs):

            callback.on_epoch_begin(epoch)

            # Set model in training mode
            self.model.train(True)

            # Initialize batch objective accumulator
            batch_objective = []

            callback.on_step_begin(self.data.get_train_size(),
                                   log_metric_history)

            for i, (x, y) in enumerate(train_loader):

                callback.on_batch_begin(i, x.size(0))

                # Prepare minibatch
                if self.use_cuda:
                    x, y = x.cuda(), y.cuda()

                # Normalize x and y
                if self.normalize_x:
                    x = (x - self.x_means) / self.x_stds
                if self.normalize_y:
                    y = (y - self.y_mean) / self.y_std

                # Update parameters
                def closure():
                    self.optimizer.zero_grad()
                    logits = self.prediction(x)
                    loss = self.objective(logits, y)
                    loss.backward()
                    return loss

                loss = self.optimizer.step(closure)

                callback.on_batch_end()

                if log_metric_history and (i + 1) in eval_iters:

                    # Set model in test mode
                    self.model.train(False)

                    # Evaluate model
                    with torch.no_grad():
                        self._evaluate_model(
                            self.metric_history, x_train, y_train, x_test, y_test
                        )

                # Store batch objective
                batch_objective.append(loss.detach().cpu().item())

            # Compute and store average objective from last epoch
            self.objective_history.append(np.mean(batch_objective))

            callback.on_step_end()

            if log_metric_history:

                # Print progress
                msg = self._print_progress(epoch)

            else:

                # Print average objective from last epoch
                msg = self._print_objective(epoch)

            callback.on_epoch_end(msg)

        # Set model in test mode
        self.model.train(False)

        callback.on_end()

        # Evaluate model
        with torch.no_grad():
            self._evaluate_model(self.final_metric, x_train, y_train, x_test, y_test)


    def _evaluate_model(self, metric_dict, x_train, y_train, x_test, y_test):

        ## All subclasses should override:
        raise NotImplementedError

    def _print_progress(self, epoch):

        ## All subclasses should override:
        raise NotImplementedError

    def _print_objective(self, epoch):

        # Print average objective from last epoch
        msg = [
            "Dataset: {:12s}, Epoch [{:2d}/{:2d}], Loss: {:.4f} ".format(
                self.data_params["data_set"],
                epoch + 1,
                self.train_params["num_epochs"],
                self.objective_history[-1],
            )
        ]
        return msg

    def save(
        self,
        save_final_metric=True,
        save_metric_history=True,
        save_objective_history=True,
        save_model=True,
        save_optimizer=True,
        create_folder=True,
        folder_path=None,
    ):

        # Define folder path
        if not folder_path:
            folder_path = self.folder_name

        # Create folder
        if create_folder:
            os.makedirs(folder_path, exist_ok=True)

        # Store state dictionaries for model and optimizer
        if save_model:
            torch.save(self.model.state_dict(),
                       os.path.join(folder_path, "model.pt")
            )
        if save_optimizer:
            torch.save(
                self.optimizer.state_dict(),
                os.path.join(folder_path, "optimizer.pt")
            )

        # Store history
        if save_final_metric:
            output = open(os.path.join(folder_path, "final_metric.pkl"), "wb")
            pickle.dump(self.final_metric, output)
            output.close()
        if save_metric_history:
            output = open(os.path.join(folder_path, "metric_history.pkl"), "wb")
            pickle.dump(self.metric_history, output)
            output.close()
        if save_objective_history:
            output = open(os.path.join(folder_path, "objective_history.pkl"), "wb")
            pickle.dump(self.objective_history, output)
            output.close()

    def load(
        self,
        load_final_metric=True,
        load_metric_history=True,
        load_objective_history=True,
        load_model=True,
        load_optimizer=True,
        folder_path=None,
    ):

        # Define folder path
        if not folder_path:
            folder_path = self.folder_name
        # Load state dictionaries for model and optimizer
        if load_model:
            state_dict = torch.load(os.path.join(folder_path, "model.pt"))
            self.model.load_state_dict(state_dict)
            self.model.train(False)
        if load_optimizer:
            state_dict = torch.load(os.path.join(folder_path, "optimizer.pt"))
            self.optimizer.load_state_dict(state_dict)

        # Load history
        if load_final_metric:
            pkl_file = open(os.path.join(folder_path, "final_metric.pkl"), "rb")
            self.final_metric = pickle.load(pkl_file)
            pkl_file.close()
        if load_metric_history:
            pkl_file = open(os.path.join(folder_path, "metric_history.pkl"), "rb")
            self.metric_history = pickle.load(pkl_file)
            pkl_file.close()
        if load_objective_history:
            pkl_file = open(os.path.join(folder_path, "objective_history.pkl"), "rb")
            self.objective_history = pickle.load(pkl_file)
            pkl_file.close()


#####################################
## Define experiment class for BBB ##
#####################################


class ExperimentBBBMLPReg(Experiment):
    def __init__(
        self,
        data_set,
        model_params,
        train_params,
        optim_params,
        evals_per_epoch=1,
        normalize_x=False,
        normalize_y=False,
        results_folder="./results",
        data_folder=DEFAULT_DATA_FOLDER,
        use_cuda=torch.cuda.is_available(),
    ):
        super(type(self), self).__init__(
            data_set,
            model_params,
            train_params,
            optim_params,
            evals_per_epoch,
            normalize_x,
            normalize_y,
            results_folder,
            data_folder,
            use_cuda,
        )

        # Define name for experiment class
        experiment_name = "bbb_mlp_reg"

        # Define folder name for results
        self.folder_name = folder_name(
            experiment_name,
            data_set,
            model_params,
            train_params,
            optim_params,
            results_folder,
        )

        # Initialize model
        self.model = BNN(
            input_size=self.data.num_features,
            hidden_sizes=model_params["hidden_sizes"],
            output_size=self.data.num_classes,
            act_func=model_params["act_func"],
            prior_prec=model_params["prior_prec"],
            prec_init=optim_params["prec_init"],
        )
        if use_cuda:
            self.model = self.model.cuda()

        # Define prediction function
        def prediction(x):
            mu_list = [
                self.model(x) for _ in range(self.train_params["train_mc_samples"])
            ]
            return mu_list

        self.prediction = prediction

        # Define objective
        def objective(mu_list, y):
            return metrics.avneg_elbo_gaussian(
                mu_list,
                y,
                tau=self.model_params["noise_prec"],
                train_set_size=self.data.get_train_size(),
                kl=self.model.kl_divergence(),
            )

        self.objective = objective

        # Initialize optimizer
        self.optimizer = Adam(
            self.model.parameters(),
            lr=optim_params["learning_rate"],
            betas=optim_params["betas"],
            eps=1e-8,
        )

        # Initialize metric history
        self.metric_history = dict(
            elbo_neg_ave=[],
            train_pred_logloss=[],
            train_pred_rmse=[],
            test_pred_logloss=[],
            test_pred_rmse=[],
        )

        # Initialize final metric
        self.final_metric = dict(
            elbo_neg_ave=[],
            train_pred_logloss=[],
            train_pred_rmse=[],
            test_pred_logloss=[],
            test_pred_rmse=[],
        )

    def _evaluate_model(self, metric_dict, x_train, y_train, x_test, y_test):

        # Unnormalize noise precision
        if self.normalize_y:
            tau = self.model_params["noise_prec"] / (self.y_std ** 2)
        else:
            tau = self.model_params["noise_prec"]

        # Normalize train x
        if self.normalize_x:
            x_train = (x_train - self.x_means) / self.x_stds

        # Get train predictions
        mu_list = [
            self.model(x_train) for _ in range(self.train_params["eval_mc_samples"])
        ]

        # Unnormalize train predictions
        if self.normalize_y:
            mu_list = [self.y_mean + self.y_std * mu for mu in mu_list]

        # Store train metrics
        metric_dict["train_pred_logloss"].append(
            metrics.predictive_avneg_loglik_gaussian(mu_list, y_train, tau=tau)
            .detach()
            .cpu()
            .item()
        )
        metric_dict["train_pred_rmse"].append(
            metrics.predictive_rmse(mu_list, y_train).detach().cpu().item()
        )
        metric_dict["elbo_neg_ave"].append(
            metrics.avneg_elbo_gaussian(
                mu_list,
                y_train,
                tau=tau,
                train_set_size=self.data.get_train_size(),
                kl=self.model.kl_divergence(),
            )
            .detach()
            .cpu()
            .item()
        )

        # Normalize test x
        if self.normalize_x:
            x_test = (x_test - self.x_means) / self.x_stds

        # Get test predictions
        mu_list = [
            self.model(x_test) for _ in range(self.train_params["eval_mc_samples"])
        ]

        # Unnormalize test predictions
        if self.normalize_y:
            mu_list = [self.y_mean + self.y_std * mu for mu in mu_list]

        # Store test metrics
        metric_dict["test_pred_logloss"].append(
            metrics.predictive_avneg_loglik_gaussian(mu_list, y_test, tau=tau)
            .detach()
            .cpu()
            .item()
        )
        metric_dict["test_pred_rmse"].append(
            metrics.predictive_rmse(mu_list, y_test).detach().cpu().item()
        )

    def _print_progress(self, epoch):

        # Print progress
        msg = [
            "Epoch [{}/{}], Neg. Ave. ELBO: {:.4f}, Logloss: {:.4f}, Test Logloss: {:.4f} ".format(
                epoch + 1,
                self.train_params["num_epochs"],
                self.metric_history["elbo_neg_ave"][-1],
                self.metric_history["train_pred_logloss"][-1],
                self.metric_history["test_pred_logloss"][-1],
            )
        ]
        return msg


##################################################
## Define experiment class for Bayesian Dropout ##
##################################################


class ExperimentDropoutMLPReg(Experiment):
    def __init__(
        self,
        data_set,
        model_params,
        train_params,
        optim_params,
        evals_per_epoch=1,
        normalize_x=False,
        normalize_y=False,
        results_folder="./results",
        data_folder=DEFAULT_DATA_FOLDER,
        use_cuda=torch.cuda.is_available(),
    ):
        super(type(self), self).__init__(
            data_set,
            model_params,
            train_params,
            optim_params,
            evals_per_epoch,
            normalize_x,
            normalize_y,
            results_folder,
            data_folder,
            use_cuda,
        )

        # Define name for experiment class
        experiment_name = "dropout_mlp_reg"

        # Define folder name for results
        self.folder_name = folder_name(
            experiment_name,
            data_set,
            model_params,
            train_params,
            optim_params,
            results_folder,
        )

        # Initialize model
        self.model = DropoutMLP(
            input_size=self.data.num_features,
            hidden_sizes=model_params["hidden_sizes"],
            output_size=self.data.num_classes,
            act_func=model_params["act_func"],
            prior_prec=model_params["prior_prec"],
            drop_prob=model_params["dropout"],
        )
        if self.use_cuda:
            self.model = self.model.cuda()

        # Define prediction function
        def prediction(x):
            mu_list = [
                self.model(x) for _ in range(self.train_params["train_mc_samples"])
            ]
            return mu_list

        self.prediction = prediction

        # Define objective ?
        def objective(mu_list, y):
            return metrics.mc_mse(mu_list, y)
            # return metrics.avneg_elbo_gaussian(
            #   mu_list, y, tau=self.model_params['noise_prec'],
            #   train_set_size = self.data.get_current_train_size(),
            #   kl = self.model.kl_divergence()
            # )

        self.objective = objective

        # Initialize optmizer
        tau = model_params["noise_prec"]
        N = self.data.get_train_size()
        weight_decay = (
            (1 - model_params["dropout"])
            * (model_params["prior_prec"] ** 2)
            / (2 * tau * N)
        )

        self.optimizer = Adam(
            self.model.parameters(),
            lr=optim_params["learning_rate"],
            betas=optim_params["betas"],
            weight_decay=weight_decay,
            eps=1e-8,
        )
        # Initialize metric history
        self.metric_history = dict(
            mse_ave=[],  # Shouldn't it be the rmse that its optimized?
            train_pred_logloss=[],
            train_pred_rmse=[],
            test_pred_logloss=[],
            test_pred_rmse=[],
        )

        # Initialize final metric
        self.final_metric = dict(
            mse_ave=[],  # Shouldn't it be the rmse that its optimized?
            train_pred_logloss=[],
            train_pred_rmse=[],
            test_pred_logloss=[],
            test_pred_rmse=[],
        )

    def _evaluate_model(self, metric_dict, x_train, y_train, x_test, y_test):

        # Unnormalize noise precision
        if self.normalize_y:
            tau = self.model_params["noise_prec"] / (self.y_std ** 2)
        else:
            tau = self.model_params["noise_prec"]

        # Normalize train x
        if self.normalize_x:
            x_train = (x_train - self.x_means) / self.x_stds

        # Get train predictions
        mu_list = [
            self.model(x_train) for _ in range(self.train_params["eval_mc_samples"])
        ]

        # Unnormalize train predictions
        if self.normalize_y:
            mu_list = [self.y_mean + self.y_std * mu for mu in mu_list]

        # Store train metrics
        metric_dict["train_pred_logloss"].append(
            metrics.predictive_avneg_loglik_gaussian(mu_list, y_train, tau=tau)
            .detach()
            .cpu()
            .item()
        )
        metric_dict["train_pred_rmse"].append(
            metrics.predictive_rmse(mu_list, y_train).detach().cpu().item()
        )
        metric_dict["mse_ave"].append(
            metrics.mc_mse(mu_list, y_train).detach().cpu().item()
        )

        # Normalize test x
        if self.normalize_x:
            x_test = (x_test - self.x_means) / self.x_stds

        # Get test predictions
        mu_list = [
            self.model(x_test) for _ in range(self.train_params["eval_mc_samples"])
        ]

        # Unnormalize test predictions
        if self.normalize_y:
            mu_list = [self.y_mean + self.y_std * mu for mu in mu_list]

        # Store test metrics
        metric_dict["test_pred_logloss"].append(
            metrics.predictive_avneg_loglik_gaussian(mu_list, y_test, tau=tau)
            .detach()
            .cpu()
            .item()
        )
        metric_dict["test_pred_rmse"].append(
            metrics.predictive_rmse(mu_list, y_test).detach().cpu().item()
        )

    def _print_progress(self, epoch):

        # Print progress
        msg = [
            "Epoch [{}/{}], Ave. MSE: {:.4f}, Logloss: {:.4f}, Test Logloss: {:.4f} ".format(
                epoch + 1,
                self.train_params["num_epochs"],
                self.metric_history["mse_ave"][-1],
                self.metric_history["train_pred_logloss"][-1],
                self.metric_history["test_pred_logloss"][-1],
            )
        ]
        return msg


#######################################
## Define experiment class for Vadam ##
#######################################


class ExperimentVadamMLPReg(Experiment):
    def __init__(
        self,
        data_set,
        model_params,
        train_params,
        optim_params,
        evals_per_epoch=1,
        normalize_x=False,
        normalize_y=False,
        results_folder="./results",
        data_folder=DEFAULT_DATA_FOLDER,
        use_cuda=torch.cuda.is_available(),
    ):
        super(type(self), self).__init__(
            data_set,
            model_params,
            train_params,
            optim_params,
            evals_per_epoch,
            normalize_x,
            normalize_y,
            results_folder,
            data_folder,
            use_cuda,
        )

        # Define name for experiment class
        experiment_name = "vadam_mlp_reg"

        # Define folder name for results
        self.folder_name = folder_name(
            experiment_name,
            data_set,
            model_params,
            train_params,
            optim_params,
            results_folder,
        )

        # Initialize model
        self.model = MLP(
            input_size=self.data.num_features,
            hidden_sizes=model_params["hidden_sizes"],
            output_size=self.data.num_classes,
            act_func=model_params["act_func"],
        )
        if use_cuda:
            self.model = self.model.cuda()

        # Define prediction function
        def prediction(x):
            mu = self.model(x)
            return mu

        self.prediction = prediction

        # Define objective
        def objective(mu, y):
            return metrics.avneg_loglik_gaussian(
                mu, y, tau=self.model_params["noise_prec"]
            )

        self.objective = objective

        # Initialize optimizer
        self.optimizer = Vadam(
            self.model.parameters(),
            lr=optim_params["learning_rate"],
            betas=optim_params["betas"],
            prior_prec=model_params["prior_prec"],
            prec_init=optim_params["prec_init"],
            num_samples=train_params["train_mc_samples"],
            train_set_size=self.data.get_train_size(),
        )

        # Initialize metric history
        self.metric_history = dict(
            elbo_neg_ave=[],
            train_pred_logloss=[],
            train_pred_rmse=[],
            test_pred_logloss=[],
            test_pred_rmse=[],
        )

        # Initialize final metric
        self.final_metric = dict(
            elbo_neg_ave=[],
            train_pred_logloss=[],
            train_pred_rmse=[],
            test_pred_logloss=[],
            test_pred_rmse=[],
        )

    def _evaluate_model(self, metric_dict, x_train, y_train, x_test, y_test):

        # Unnormalize noise precision
        if self.normalize_y:
            tau = self.model_params["noise_prec"] / (self.y_std ** 2)
        else:
            tau = self.model_params["noise_prec"]

        # Normalize train x
        if self.normalize_x:
            x_train = (x_train - self.x_means) / self.x_stds

        # Get train predictions
        mu_list = self.optimizer.get_mc_predictions(
            self.model.forward,
            inputs=x_train,
            mc_samples=self.train_params["eval_mc_samples"],
            ret_numpy=False,
        )

        # Unnormalize train predictions
        if self.normalize_y:
            mu_list = [self.y_mean + self.y_std * mu for mu in mu_list]

        # Store train metrics
        metric_dict["train_pred_logloss"].append(
            metrics.predictive_avneg_loglik_gaussian(mu_list, y_train, tau=tau)
            .detach()
            .cpu()
            .item()
        )
        metric_dict["train_pred_rmse"].append(
            metrics.predictive_rmse(mu_list, y_train).detach().cpu().item()
        )
        metric_dict["elbo_neg_ave"].append(
            metrics.avneg_elbo_gaussian(
                mu_list,
                y_train,
                tau=tau,
                train_set_size=self.data.get_train_size(),
                kl=self.optimizer.kl_divergence(),
            )
            .detach()
            .cpu()
            .item()
        )

        # Normalize test x
        if self.normalize_x:
            x_test = (x_test - self.x_means) / self.x_stds

        # Get test predictions
        mu_list = self.optimizer.get_mc_predictions(
            self.model.forward,
            inputs=x_test,
            mc_samples=self.train_params["eval_mc_samples"],
            ret_numpy=False,
        )

        # Unnormalize test predictions
        if self.normalize_y:
            mu_list = [self.y_mean + self.y_std * mu for mu in mu_list]

        # Store test metrics
        metric_dict["test_pred_logloss"].append(
            metrics.predictive_avneg_loglik_gaussian(mu_list, y_test, tau=tau)
            .detach()
            .cpu()
            .item()
        )
        metric_dict["test_pred_rmse"].append(
            metrics.predictive_rmse(mu_list, y_test).detach().cpu().item()
        )

    def _print_progress(self, epoch):

        # Print progress
        msg = [
            "Epoch [{}/{}], Neg. Ave. ELBO: {:.4f}, Logloss: {:.4f}, Test Logloss: {:.4f} ".format(
                epoch + 1,
                self.train_params["num_epochs"],
                self.metric_history["elbo_neg_ave"][-1],
                self.metric_history["train_pred_logloss"][-1],
                self.metric_history["test_pred_logloss"][-1],
            )
        ]
        return msg


#######################################
## Define experiment class for Vprop ##
#######################################


class ExperimentVpropMLPReg(Experiment):
    def __init__(
        self,
        data_set,
        model_params,
        train_params,
        optim_params,
        evals_per_epoch=1,
        normalize_x=False,
        normalize_y=False,
        results_folder="./results",
        data_folder=DEFAULT_DATA_FOLDER,
        use_cuda=torch.cuda.is_available(),
    ):
        super(type(self), self).__init__(
            data_set,
            model_params,
            train_params,
            optim_params,
            evals_per_epoch,
            normalize_x,
            normalize_y,
            results_folder,
            data_folder,
            use_cuda,
        )

        # Define name for experiment class
        experiment_name = "vprop_mlp_reg"

        # Define folder name for results
        self.folder_name = folder_name(
            experiment_name,
            data_set,
            model_params,
            train_params,
            optim_params,
            results_folder,
        )

        # Initialize model
        self.model = MLP(
            input_size=self.data.num_features,
            hidden_sizes=model_params["hidden_sizes"],
            output_size=self.data.num_classes,
            act_func=model_params["act_func"],
        )
        if use_cuda:
            self.model = self.model.cuda()

        # Define prediction function
        def prediction(x):
            mu = self.model(x)
            return mu

        self.prediction = prediction

        # Define objective
        def objective(mu, y):
            return metrics.avneg_loglik_gaussian(
                mu, y, tau=self.model_params["noise_prec"]
            )

        self.objective = objective

        # Initialize optimizer
        self.optimizer = Vprop(
            self.model.parameters(),
            lr=optim_params["learning_rate"],
            beta=optim_params["beta"],
            prior_prec=model_params["prior_prec"],
            prec_init=optim_params["prec_init"],
            num_samples=train_params["train_mc_samples"],
            train_set_size=self.data.get_train_size(),
        )

        # Initialize metric history
        self.metric_history = dict(
            elbo_neg_ave=[],
            train_pred_logloss=[],
            train_pred_rmse=[],
            test_pred_logloss=[],
            test_pred_rmse=[],
        )

        # Initialize final metric
        self.final_metric = dict(
            elbo_neg_ave=[],
            train_pred_logloss=[],
            train_pred_rmse=[],
            test_pred_logloss=[],
            test_pred_rmse=[],
        )

    def _evaluate_model(self, metric_dict, x_train, y_train, x_test, y_test):

        # Unnormalize noise precision
        if self.normalize_y:
            tau = self.model_params["noise_prec"] / (self.y_std ** 2)
        else:
            tau = self.model_params["noise_prec"]

        # Normalize train x
        if self.normalize_x:
            x_train = (x_train - self.x_means) / self.x_stds

        # Get train predictions
        mu_list = self.optimizer.get_mc_predictions(
            self.model.forward,
            inputs=x_train,
            mc_samples=self.train_params["eval_mc_samples"],
            ret_numpy=False,
        )

        # Unnormalize train predictions
        if self.normalize_y:
            mu_list = [self.y_mean + self.y_std * mu for mu in mu_list]

        # Store train metrics
        metric_dict["train_pred_logloss"].append(
            metrics.predictive_avneg_loglik_gaussian(mu_list, y_train, tau=tau)
            .detach()
            .cpu()
            .item()
        )
        metric_dict["train_pred_rmse"].append(
            metrics.predictive_rmse(mu_list, y_train).detach().cpu().item()
        )
        metric_dict["elbo_neg_ave"].append(
            metrics.avneg_elbo_gaussian(
                mu_list,
                y_train,
                tau=tau,
                train_set_size=self.data.get_train_size(),
                kl=self.optimizer.kl_divergence(),
            )
            .detach()
            .cpu()
            .item()
        )

        # Normalize test x
        if self.normalize_x:
            x_test = (x_test - self.x_means) / self.x_stds

        # Get test predictions
        mu_list = self.optimizer.get_mc_predictions(
            self.model.forward,
            inputs=x_test,
            mc_samples=self.train_params["eval_mc_samples"],
            ret_numpy=False,
        )

        # Unnormalize test predictions
        if self.normalize_y:
            mu_list = [self.y_mean + self.y_std * mu for mu in mu_list]

        # Store test metrics
        metric_dict["test_pred_logloss"].append(
            metrics.predictive_avneg_loglik_gaussian(mu_list, y_test, tau=tau)
            .detach()
            .cpu()
            .item()
        )
        metric_dict["test_pred_rmse"].append(
            metrics.predictive_rmse(mu_list, y_test).detach().cpu().item()
        )

    def _print_progress(self, epoch):

        # Print progress
        msg = [
            "Epoch [{}/{}], Neg. Ave. ELBO: {:.4f}, Logloss: {:.4f}, Test Logloss: {:.4f} ".format(
                epoch + 1,
                self.train_params["num_epochs"],
                self.metric_history["elbo_neg_ave"][-1],
                self.metric_history["train_pred_logloss"][-1],
                self.metric_history["test_pred_logloss"][-1],
            )
        ]
        return msg
