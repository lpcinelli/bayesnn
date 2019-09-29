import gzip
import pickle
import sys

import numpy as np

from . import pbp

sys.setrecursionlimit(10000)


class PBP_net:
    def __init__(self, input_size, hidden_sizes, output_size=1):
        """
            Constructor for the class implementing a Bayesian neural network
            trained with the probabilistic back propagation method.

            @param input_size   Number of different features for the training data.

            @param hidden_sizes Vector with the number of neurons for each
                                hidden layer.
            @param n_epochs     Numer of epochs for which to train the
                                network. The recommended value 40 should be
                                enough.
        """

        # We construct the network
        if output_size != 1:
            raise ValueError(
                "Current PBP implementation only works for regression with output size equals 1"
            )

        n_units_per_layer = np.concatenate(
            ([input_size], hidden_sizes, [output_size]))
        self.pbp_instance = pbp.PBP(n_units_per_layer)

    def set_y_mean_std(self, mean_y_train, std_y_train):
        self.pbp_instance.set_y_mean_std(mean_y_train, std_y_train)

    def set_x_mean_std(self, mean_x_train, std_x_train):
        self.mean_X_train = mean_x_train
        self.std_X_train = std_x_train

    def step(self, X_train, y_train, n_epochs=1):
        """
            Function that re-trains the network on some data.

            @param X_train      Matrix with the features for the training data.
            @param y_train      Vector with the target variables for the
                                training data.
            @param n_epochs     Numer of epochs for which to train the
                                network.
        """

        # X_train and  y_train MUST be normalized
        self.pbp_instance.do_pbp(X_train, y_train, n_epochs)

        return self.pbp_instance.logZ

    def predict(self, X_test):
        """
            Function for making predictions with the Bayesian neural network.

            @param X_test   The matrix of features for the test data


            @return m       The predictive mean for the test target variables.
            @return v       The predictive variance for the test target
                            variables.
            @return v_noise The estimated variance for the additive noise.

        """

        # X_test MUST be normalized (this is done within the uci-code.Experiment.run())
        X_test = np.array(X_test, ndmin=2)

        # We compute the predictive mean and variance for the target variables
        # of the test data

        m, v, v_noise = self.pbp_instance.get_predictive_mean_and_variance(
            X_test)

        # We are done!

        return m, v, v_noise

    def predict_deterministic(self, X_test):
        """
            Function for making predictions with the Bayesian neural network.

            @param X_test   The matrix of features for the test data


            @return o       The predictive value for the test target variables.

        """

        X_test = np.array(X_test, ndmin=2)

        # this is done within the uci-code.Experiment.run()
        # We normalize the test set
        # X_test = (X_test - np.full(X_test.shape, self.mean_X_train)) / np.full(
        #     X_test.shape, self.std_X_train
        # )

        # We compute the predictive mean and variance for the target variables
        # of the test data
        o = self.pbp_instance.get_deterministic_output(X_test)

        # We are done!
        return o

    def sample_weights(self):
        """
            Function that draws a sample from the posterior approximation
            to the weights distribution.

        """

        self.pbp_instance.sample_w()

    def save_to_file(self, filename):
        """
            Function that stores the network in a file.

            @param filename   The name of the file.

        """

        # We save the network to a file using pickle

        def save_object(obj, filename):

            result = pickle.dumps(obj)
            with gzip.GzipFile(filename, "wb") as dest:
                dest.write(result)
            dest.close()

        save_object(self, filename)


def load_PBP_net_from_file(filename):
    """
        Function that load a network from a file.

        @param filename   The name of the file.

    """
    def load_object(filename):

        with gzip.GzipFile(filename, "rb") as source:
            result = source.read()
        ret = pickle.loads(result)
        source.close()

        return ret

    # We load the dictionary with the network parameters

    PBP_network = load_object(filename)

    return PBP_network
