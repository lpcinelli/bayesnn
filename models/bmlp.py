import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from ..metrics import kl_gaussian, stoch_kl_gaussian
from torch.distributions import Normal


def select_act(act_func):
    if act_func == "relu":
        return F.relu
    if act_func == "tanh":
        return F.tanh
    if act_func == "sigmoid":
        return F.sigmoid
    raise ValueError("Invalid activation func. (relu, tanh, sigmoid).")

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
        obs_noise_init=1.0,
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

        # Set observation noise (tuned as precision parameter of gaussian dist.)
        self.log_noise = Parameter(
            torch.Tensor(self.output_size).fill_(math.log(obs_noise_init))
        )

        # Set activation function
        self.act = select_act(act_func)

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
        """ Init rule for the mean (torch.nn.init.kaiming_uniform_ as in
        torch.nn.Linear): W ~ U[ -1/fan_in, 1/fan_in ];

        Using this initialization scheme (normal r.v. w/ uniform mean) the
        resulting distribution is Gaussian if `stdv' <= `sigma_init`, else it
        gets flat on top as a plateau. The variance of resulting random
        variable is :math: `var(Uni) + var(Normal) = 1/(3*fan_in^2) +
        sigma_init^2`. As `prec_init` is to 10 and :math: `sigma_init^2 =
        1.0/prec_init = 0.1`, it dominates over the `fan_in` term and the final
        r.v. is Gaussian. Hence, we have a Gaussian init :math: `N(0,
        1/prec_init)`.
        """
        # self.weight_mu.data.uniform_(-0.05, 0.05)
        # self.weight_spsigma.data.uniform_(-2, -1)

        # self.bias_mu.data.uniform_(-0.05, 0.05)
        # self.bias_spsigma.data.uniform_(-2, -1)

        stdv = 1.0 / math.sqrt(self.weight_mu.size(1))

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

        # self.weight = Normal(self.weight_mu, F.softplus(self.weight_spsigma)).rsample()
        weight = self.transformation(self.weight_mu, self.weight_spsigma)
        if self.bias is not None:
            bias = self.transformation(self.bias_mu, self.bias_spsigma)
            # self.bias = Normal(self.bias_mu, self.bias_spsigma).rsample()

        return F.linear(input, weight, bias)

    def kl_divergence(self):
        """Compute the analytical KL divergence between current distribution and
        the prior. We assume the prior is a (multivariate) standard gaussian.

        Returns:
            [type] -- [description]
        """

        kl = kl_gaussian(
            q_mu=self.weight_mu,
            q_sigma=F.softplus(self.weight_spsigma),
            p_mu=torch.zeros_like(self.weight_mu),
            p_sigma=torch.ones_like(self.weight_spsigma) * self.sigma_prior,
        )
        if self.bias is not None:
            kl += kl_gaussian(
                q_mu=self.bias_mu,
                q_sigma=F.softplus(self.bias_spsigma),
                p_mu=torch.zeros_like(self.bias_mu),
                p_sigma=torch.ones_like(self.bias_spsigma) * self.sigma_prior,
            )
        return kl

    # def stoch_kl_divergence(self):
    #     """Compute the analytical KL divergence between current distribution and
    #     the prior. We assume the prior is a (multivariate) standard gaussian.

    #     Returns:
    #         [type] -- [description]
    #     """
    #     kl_approx = stoch_kl_gaussian(samples=self.weight,
    #                         q_mu=self.weight_mu,
    #                         q_sigma=F.softplus(self.weight_spsigma),
    #                         p_mu=torch.zeros_like(self.weight_mu),
    #                         p_sigma=torch.ones_like(self.weight_spsigma)*self.sigma_prior
    #                         )

    #     if self.bias is not None:
    #         kl_approx += stoch_kl_gaussian(
    #                         samples=self.bias,
    #                         q_mu=self.bias_mu,
    #                         q_sigma=F.softplus(self.bias_spsigma),
    #                         p_mu=torch.zeros_like(self.bias_mu),
    #                         p_sigma=torch.ones_like(self.bias_spsigma)*self.sigma_prior
    #                         )
    #     return kl_approx

    def extra_repr(self):
        return "in_features={}, out_features={}, sigma_prior={}, sigma_init={}, bias={}".format(
            self.in_features,
            self.out_features,
            self.sigma_prior,
            self.sigma_init,
            self.bias is not None,
        )
