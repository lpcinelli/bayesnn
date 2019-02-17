import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .metrics import kl_gaussian, stoch_kl_gaussian
from torch.distributions import Normal

############################
## Multi-Layer Perceptron ##
############################


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, act_func="relu"):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        if output_size is not None:
            self.output_size = output_size
            self.squeeze_output = False
        else:
            self.output_size = 1
            self.squeeze_output = True

        # Set activation function
        if act_func == "relu":
            self.act = F.relu
        elif act_func == "tanh":
            self.act = F.tanh
        elif act_func == "sigmoid":
            self.act = F.sigmoid

        # Define layers
        if len(hidden_sizes) == 0:
            # Linear model
            self.hidden_layers = []
            self.output_layer = nn.Linear(self.input_size, self.output_size)
        else:
            # Neural network
            self.hidden_layers = nn.ModuleList(
                [
                    nn.Linear(in_size, out_size)
                    for in_size, out_size in zip(
                        [self.input_size] + hidden_sizes[:-1], hidden_sizes
                    )
                ]
            )
            self.output_layer = nn.Linear(hidden_sizes[-1], self.output_size)

    def forward(self, x):
        # vectorize input samples
        x = x.view(-1, self.input_size)
        out = x
        for layer in self.hidden_layers:
            out = self.act(layer(out))
        z = self.output_layer(out)
        if self.squeeze_output:
            z = torch.squeeze(z).view([-1])
        return z


#############################
## Bayesian Neural Network ##
#############################


class BNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_sizes,
        output_size,
        act_func="relu",
        prior_prec=1.0,
        prec_init=1.0,
    ):
        super(type(self), self).__init__()
        self.input_size = input_size
        sigma_prior = 1.0 / math.sqrt(prior_prec)
        sigma_init = 1.0 / math.sqrt(prec_init)
        if output_size:
            self.output_size = output_size
            self.squeeze_output = False
        else:
            self.output_size = 1
            self.squeeze_output = True

        # Set activation function
        if act_func == "relu":
            self.act = F.relu
        elif act_func == "tanh":
            self.act = F.tanh
        elif act_func == "sigmoid":
            self.act = F.sigmoid

        # Define layers
        if len(hidden_sizes) == 0:
            # Linear model
            self.hidden_layers = []
            self.output_layer = StochasticLinear(
                self.input_size,
                self.output_size,
                sigma_prior=sigma_prior,
                sigma_init=sigma_init,
            )
        else:
            # Neural network
            self.hidden_layers = nn.ModuleList(
                [
                    StochasticLinear(
                        in_size,
                        out_size,
                        sigma_prior=sigma_prior,
                        sigma_init=sigma_init,
                    )
                    for in_size, out_size in zip(
                        [self.input_size] + hidden_sizes[:-1], hidden_sizes
                    )
                ]
            )
            self.output_layer = StochasticLinear(
                hidden_sizes[-1],
                self.output_size,
                sigma_prior=sigma_prior,
                sigma_init=sigma_init,
            )

    def forward(self, x):
        x = x.view(-1, self.input_size)
        out = x
        for layer in self.hidden_layers:
            out = self.act(layer(out))
        z = self.output_layer(out)
        if self.squeeze_output:
            z = torch.squeeze(z).view([-1])
        return z

    def kl_divergence(self):
        kl = 0

        for layer in self.hidden_layers:
            kl += layer.kl_divergence()
        kl += self.output_layer.kl_divergence()
        return kl


###############################################
## Gaussian Mean-Field Linear Transformation ##
###############################################


class StochasticLinear(nn.Module):
    """Applies a stochastic linear transformation to the incoming data:
    :math:`y = Ax + b`. This is a stochastic variant of the in-built
    torch.nn.Linear(). Stochasticity stems from independent gaussian weights
    (mean-field assumption), that is, w_i ~ N(mu_i, sigma_i), thus this layer
    doubles the number of learnable params of its deterministic counterpart.

    The layer applies the reparameterisation trick (pathwise derivative estimator):
    :math:`w_i = mu_i + sigma_i*N(0,1)` to enable the insertion and computation
    of random nodes in the computational graph. The underling assumption is that
    the both the model and the random variable transformation are differentiable.

    The layer exposes a closed-form analytically computed Kullback-Leibler
    divergence method (`kl_divergence`) between the approximating distribution
    and the prior (gaussian centered at zero), as in (Graves, 2011). For complex
    distributions use `stoch_kl_divergence` that computes the KL as the difference
    between the log-likelihoods of the samples. These methods are useful when
    computing model complexity losses.

    """

    def __init__(
        self, in_features, out_features, sigma_prior=1.0, sigma_init=1.0, bias=True
    ):
        super(type(self), self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_prior = sigma_prior
        self.sigma_init = sigma_init
        self.weight_mu = Parameter(torch.Tensor(out_features, in_features))
        self.weight_spsigma = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = True
            self.bias_mu = Parameter(torch.Tensor(out_features))
            self.bias_spsigma = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        # Init rule (for the mean): W ~ U[ -1/fan_in, 1/fan_in ]
        stdv = 1. / math.sqrt(self.weight_mu.size(1))

        # Why not directly initialize weights_mu = 0 ; weight_spsigma = stdv ?
        self.weight_mu.data.uniform_(-stdv, stdv)

        # rho = log(exp(sigma) - 1) -> inverse of softplus transform
        self.weight_spsigma.data.fill_(math.log(math.exp(self.sigma_init) - 1))

        if self.bias is not None:
            self.bias_mu.data.uniform_(-stdv, stdv)
            self.bias_spsigma.data.fill_(math.log(math.exp(self.sigma_init) - 1))

    @staticmethod
    def transformation(mu, rho):
        # sample from base distribution: standard normal
        base_distribution = torch.normal(mean=torch.zeros_like(mu), std=1.0)

        # Apply reparametrization: w = mu + std*Normal(0,1)
        weight = mu + F.softplus(rho) * base_distribution

        return weight

    def forward(self, input):
        # Sampling in the weight space can be replaced by sampling in the
        # activation space (lower variance): Local reparameterisation trick
        # (Kingma, 2015)

        # TODO: Use torch.distribution.Normal.rsample() to instantiate a normal
        # and sample with reparameterisation
        # dist = Normal(loc=self.weight_mu, scale=F.softplus(self.weight_spsigma))

        self.weight = Normal(self.weight_mu, F.softplus(self.weight_spsigma)).rsample()
        # self.weight = self.transformation(self.weight_mu, self.weight_spsigma)
        if self.bias is not None:
            # self.bias = self.transformation(self.bias_mu, self.bias_spsigma)
            self.bias = Normal(self.bias_mu, self.bias_spsigma).rsample()

        return F.linear(input, self.weight, self.bias)

    def kl_divergence(self):
        '''Compute the analytical KL divergence between current distribution and
        the prior. We assume the prior is a (multivariate) standard gaussian.

        Returns:
            [type] -- [description]
        '''

        kl = kl_gaussian(
                q_mu=self.weight_mu,
                q_sigma=F.softplus(self.weight_spsigma),
                p_mu=torch.zeros_like(self.weight_mu),
                p_sigma=torch.ones_like(self.weight_spsigma) * self.sigma_prior
        )
        if self.bias is not None:
            kl += kl_gaussian(
                    q_mu=self.bias_mu,
                    q_sigma=F.softplus(self.bias_spsigma),
                    p_mu=torch.zeros_like(self.bias_mu),
                    p_sigma=torch.ones_like(self.bias_spsigma) * self.sigma_prior
                )
        return kl

    def stoch_kl_divergence(self):
        '''Compute the analytical KL divergence between current distribution and
        the prior. We assume the prior is a (multivariate) standard gaussian.

        Returns:
            [type] -- [description]
        '''

        mu = torch.zeros_like(self.weight_mu)
        sigma = torch.ones_like(self.weight_spsigma) * self.sigma_prior

        log_prob_qw = Normal(self.weight_mu, F.softplus(self.weight_spsigma)).log_prob(self.weight)
        log_prob_pw = Normal(mu, sigma).log_prob(self.weight)

        kl_approx = log_prob_qw.sum() - log_prob_pw.sum()
        # kl_approx = stoch_kl_gaussian(samples=self.weight,
        #                               q_mu=self.weight_mu,
        #                               q_sigma=F.softplus(self.weight_spsigma),
        #                               p_mu=mu,
        #                               p_sigma=sigma
        #                               )

        if self.bias is not None:
            mu = torch.zeros_like(self.bias_mu)
            sigma = torch.ones_like(self.bias_spsigma) * self.sigma_prior
            log_prob_qb = Normal(self.bias_mu, F.softplus(self.bias_spsigma)).log_prob(self.bias)
            log_prob_pb = Normal(mu, sigma).log_prob(self.bias)
            kl_approx += log_prob_qb.sum() - log_prob_pb.sum()
            # kl_approx += stoch_kl_gaussian(samples=self.bias,
            #                                q_mu=self.bias_mu,
            #                                q_sigma=F.softplus(self.bias_spsigma),
            #                                p_mu=mu,
            #                                p_sigma=sigma
            #                                )
        return kl_approx

    def extra_repr(self):
        return "in_features={}, out_features={}, sigma_prior={}, sigma_init={}, bias={}".format(
            self.in_features,
            self.out_features,
            self.sigma_prior,
            self.sigma_init,
            self.bias is not None,
        )


#################################################################
## MultiLayer Perceptron with support for individual gradients ##
#################################################################


class IndividualGradientMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, act_func="relu"):
        super(type(self), self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        if output_size is not None:
            self.output_size = output_size
            self.squeeze_output = False
        else:
            self.output_size = 1
            self.squeeze_output = True

        # Set activation function
        if act_func == "relu":
            self.act = F.relu
        elif act_func == "tanh":
            self.act = F.tanh
        elif act_func == "sigmoid":
            self.act = F.sigmoid

        # Define layers
        if len(hidden_sizes) == 0:
            # Linear model
            self.hidden_layers = []
            self.output_layer = nn.Linear(self.input_size, self.output_size)
        else:
            # Neural network
            self.hidden_layers = nn.ModuleList(
                [
                    nn.Linear(in_size, out_size)
                    for in_size, out_size in zip(
                        [self.input_size] + hidden_sizes[:-1], hidden_sizes
                    )
                ]
            )
            self.output_layer = nn.Linear(hidden_sizes[-1], self.output_size)

    def forward(self, x, individual_grads=False):
        """
            x: The input patterns/features.
            individual_grads: Whether or not the activations tensors and linear
                combination tensors from each layer are returned. These tensors
                are necessary for computing the GGN using goodfellow_backprop_ggn
        """

        x = x.view(-1, self.input_size)
        out = x
        # Save the model inputs, which are considered the activations of the
        # 0'th layer.
        if individual_grads:
            H_list = [out]
            Z_list = []

        for layer in self.hidden_layers:
            Z = layer(out)
            out = self.act(Z)

            # Save the activations and linear combinations from this layer.
            if individual_grads:
                H_list.append(out)
                Z.retain_grad()
                Z.requires_grad_(True)
                Z_list.append(Z)

        z = self.output_layer(out)
        if self.squeeze_output:
            z = torch.squeeze(z).view([-1])

        # Save the final model ouputs, which are the linear combinations
        # from the final layer.
        if individual_grads:
            z.retain_grad()
            z.requires_grad_(True)
            Z_list.append(z)

        if individual_grads:
            return (z, H_list, Z_list)

        return z
