import os
import pdb
import pickle

import numpy as np
import torch
from torch.optim import Adam

from .. import metrics
from ..callbacks import Timemeasure
from ..datasets import DEFAULT_DATA_FOLDER, Dataset
from ..models import PBP
from ..vadam.optimizers import Vadam, Vprop

###############################################################################
## Define function that specify folder naming convention for storing results ##
###############################################################################


def folder_name(
    experiment_name, data_set, model_params, train_params, results_folder="./results"
):

    mp = "".join(
        "{}:{}|".format(key, val) for key, val in sorted(model_params.items())
    )[:-1]
    tp = "".join(
        "{}:{}|".format(key, val) for key, val in sorted(train_params.items())
    )[:-1]
    return os.path.join(results_folder, experiment_name, data_set, mp, tp)


class ExperimentPBPReg:
    def __init__(
        self,
        data_set,
        model_params,
        train_params,
        experiment_prefix,
        normalize_x=True,
        normalize_y=True,
        results_folder="./results",
        data_folder=DEFAULT_DATA_FOLDER,
        # use_cuda=torch.cuda.is_available(),
        use_cuda=False,
        print_freq=13,
    ):

        experiment_name = "{}_pbp".format(experiment_prefix)

        # Store parameters
        self.data_set = data_set
        self.model_params = model_params
        self.train_params = train_params
        self.normalize_x = normalize_x
        self.normalize_y = normalize_y
        self.data_folder = data_folder
        self.results_folder = results_folder
        self.use_cuda = use_cuda
        self.print_freq = print_freq

        # Set random seed
        seed = train_params["seed"]
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.use_cuda:
            torch.cuda.manual_seed_all(seed)

        # Initialize metric history
        # self.objective_history = []

        # Initialize data
        self.data = Dataset(data_set=data_set, data_folder=data_folder)

        # Define folder name for results
        self.folder_name = folder_name(
            results_folder=results_folder,
            experiment_name=experiment_name,
            data_set=data_set,
            model_params=model_params,
            train_params=train_params,
        )

        # Initialize model
        # PBP inputs MUST be normalized
        self.model = PBP(
            input_size=self.data.num_features, hidden_sizes=model_params["hidden_sizes"]
        )

        # Initialize metric history
        self.metric_history = dict(
            # elbo_neg_ave=[],
            train_pred_logloss=[],
            train_pred_rmse=[],
            test_pred_logloss=[],
            test_pred_rmse=[],
        )

        # Initialize final metric
        self.final_metric = dict(
            # elbo_neg_ave=[],
            train_pred_logloss=[],
            train_pred_rmse=[],
            test_pred_logloss=[],
            test_pred_rmse=[],
        )

    def _evaluate_model(self, metric_dict, x_train, y_train, x_test, y_test):

        # Normalize train x
        if self.normalize_x:
            x_train = (x_train - self.x_means) / self.x_stds

        x_train = x_train.numpy()
        y_train = y_train.numpy()

        # Get train predictions
        m, v, v_noise = self.model.predict(x_train)

        # Denormalize test predictions
        # Model outpus are already denormalized

        # Store train metrics
        metric_dict["train_pred_logloss"].append(
            np.mean(
                -0.5 * np.log(2 * np.pi * (v + v_noise))
                - 0.5 * (y_train - m) ** 2 / (v + v_noise)
            )
        )
        metric_dict["train_pred_rmse"].append(np.sqrt(np.mean((y_train - m) ** 2)))

        # Normalize test x
        if self.normalize_x:
            x_test = (x_test - self.x_means) / self.x_stds

        x_test = x_test.numpy()
        y_test = y_test.numpy()

        # Get test predictions
        m, v, v_noise = self.model.predict(x_test)

        # Denormalize test predictions
        # Model outpus are already denormalized

        # Store test metrics
        metric_dict["test_pred_logloss"].append(
            np.mean(
                -0.5 * np.log(2 * np.pi * (v + v_noise))
                - 0.5 * (y_test - m) ** 2 / (v + v_noise)
            )
        )
        metric_dict["test_pred_rmse"].append(np.sqrt(np.mean((y_test - m) ** 2)))

    def run(self, log_metric_history=True):

        # Prepare
        callback = Timemeasure(self.data_set, self.print_freq)
        num_epochs = self.train_params["num_epochs"]
        seed = self.train_params["seed"]
        callback.on_begin()

        # Set random seed
        torch.manual_seed(seed)
        if self.use_cuda:
            torch.cuda.manual_seed_all(seed)

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


        self.model.set_y_mean_std(self.y_mean.numpy(), self.y_std.numpy())
        self.model.set_x_mean_std(self.x_means.numpy(), self.x_stds.numpy())

        # Train model
        for epoch in range(num_epochs):

            callback.on_epoch_begin(epoch)
            callback.on_step_begin(self.data.get_train_size(), log_metric_history)

            # Normalize x and y
            if self.normalize_x:
                x_train_norm = (x_train - self.x_means) / self.x_stds
            if self.normalize_y:
                y_train_norm = (y_train - self.y_mean) / self.y_std

            # # Prepare batch
            # if self.use_cuda:
            #     x_train, y_train = x_train.cuda(), y_train.cuda()

            # Update parameters
            self.model.step(x_train_norm.numpy(), y_train_norm.numpy())

            # Compute and store average objective from last epoch
            # self.objective_history.append(log_evidence)

            callback.on_step_end()

            if log_metric_history:

                # Evaluate model
                self._evaluate_model(
                    self.metric_history, x_train, y_train, x_test, y_test
                )
                # Print progress
                msg = self._print_progress(epoch)

            else:

                # Print average objective from last epoch
                msg = self._print_objective(epoch)

            callback.on_epoch_end(msg)

        callback.on_end()

        # Evaluate model
        self._evaluate_model(self.final_metric, x_train, y_train, x_test, y_test)

    def _print_progress(self, epoch):

        # Print progress
        msg = [
            "Epoch [{}/{}], Train RMSE: {:4.4f}, Test RMSE: {:4.4f}, Logloss: {:.4f}, Test Logloss: {:.4f} ".format(
                epoch + 1,
                self.train_params["num_epochs"],
                self.metric_history["train_pred_rmse"][-1],
                self.metric_history["test_pred_rmse"][-1],
                self.metric_history["train_pred_logloss"][-1],
                self.metric_history["test_pred_logloss"][-1],
            )
        ]
        return msg

    def _print_objective(self, epoch):

        # Print average objective from last epoch
        msg = [
            "Dataset: {:12s}, Epoch [{:2d}/{:2d}], Loss: {:.4f} ".format(
                self.data_params["data_set"],
                epoch + 1,
                self.train_params["num_epochs"],
                # self.objective_history[-1],
            )
        ]
        return msg

    def save(
        self,
        save_final_metric=True,
        save_metric_history=True,
        # save_objective_history=True,
        save_model=True,
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
            self.model.save_to_file(os.path.join(folder_path, "model.pt"))

        # Store history
        if save_final_metric:
            output = open(os.path.join(folder_path, "final_metric.pkl"), "wb")
            pickle.dump(self.final_metric, output)
            output.close()
        if save_metric_history:
            output = open(os.path.join(folder_path, "metric_history.pkl"), "wb")
            pickle.dump(self.metric_history, output)
            output.close()
        # if save_objective_history:
        #     output = open(os.path.join(folder_path, "objective_history.pkl"), "wb")
        #     pickle.dump(self.objective_history, output)
        #     output.close()

    def load(
        self,
        load_final_metric=True,
        load_metric_history=True,
        load_objective_history=True,
        load_model=True,
        folder_path=None,
    ):

        # Define folder path
        if not folder_path:
            folder_path = self.folder_name

        # Load state dictionaries for model and optimizer
        if load_model:
            self.model.load_PBP_net_from_file(os.path.join(folder_path, "model.pt"))

        # Load history
        if load_final_metric:
            pkl_file = open(os.path.join(folder_path, "final_metric.pkl"), "rb")
            self.final_metric = pickle.load(pkl_file)
            pkl_file.close()
        if load_metric_history:
            pkl_file = open(os.path.join(folder_path, "metric_history.pkl"), "rb")
            self.metric_history = pickle.load(pkl_file)
            pkl_file.close()
        # if load_objective_history:
        #     pkl_file = open(os.path.join(folder_path, "objective_history.pkl"), "rb")
        #     self.objective_history = pickle.load(pkl_file)
        #     pkl_file.close()
