import torch
import torch.nn as nn
import torch.nn.functional as F


def select_act(act_func):
    if act_func == "relu":
        return F.relu
    if act_func == "tanh":
        return F.tanh
    if act_func == "sigmoid":
        return F.sigmoid
    raise ValueError("Invalid activation func. (relu, tanh, sigmoid).")


##############################
## Dropout Bayesian Network ##
##############################


class DropoutMLP(nn.Module):
    """Implements "Dropout as as a Bayesian Approximation": before every weight
    layer insert a dropout layer. If using Bernoulli dropout, the underlaying
    variational distribution is

    All weight Weights are initialized by the default rule for torch.nn.Linear
    (torch.nn.init.kaiming_uniform_):
        W ~ U[ -1/fan_in, 1/fan_in ];

    Arguments:
        nn {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    def __init__(
            self,
            input_size,
            hidden_sizes,
            output_size,
            drop_prob,
            act_func="relu",
    ):
        super(type(self), self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.drop_prob = drop_prob

        # prior length-scale l -> p(W) ~ N(0, 1/l^2 * I) -> l = 1/sigma
        # reg = len_scale**2 * (1 - drop_prob) / (2. * N * tau)

        if output_size is not None:
            self.output_size = output_size
            self.squeeze_output = False
        else:
            self.output_size = 1
            self.squeeze_output = True

        # Set activation function
        self.act = select_act(act_func)

        # Define layers
        if len(hidden_sizes) == 0:
            # Linear model
            self.hidden_layers = nn.ModuleList[nn.Dropout(self.drop_prob)]
            self.output_layer = nn.Linear(self.input_size, self.output_size)
        else:
            # Neural network
            self.hidden_layers = nn.ModuleList([
                nn.Linear(in_size, out_size)
                for in_size, out_size in zip([self.input_size] +
                                             hidden_sizes[:-1], hidden_sizes)
            ])
            #     [nn.Dropout(self.drop_prob),
            #      nn.Linear(in_size, out_size)]
            #     for in_size, out_size in zip([self.input_size] +
            #                                  hidden_sizes[:-1], hidden_sizes)
            # ] + [[nn.Dropout(self.drop_prob)]]
            # self.hidden_layers = nn.ModuleList([
            #     single_layer for item in hidden_layers for single_layer in item
            # ])
            self.output_layer = nn.Linear(hidden_sizes[-1], self.output_size)

    def forward(self, x):
        # vectorize input samples
        x = x.view(-1, self.input_size)
        out = x
        for layer in self.hidden_layers:
            # out = self.act(layer(out))
            out = self.act(layer(F.dropout(out, p=self.drop_prob)))
        z = self.output_layer(F.dropout(out, p=self.drop_prob))
        if self.squeeze_output:
            z = torch.squeeze(z).view([-1])
        return z

    def train(self, mode=True):
        """Setting the mode through this function has no real effect.
        This behaviour is on purpose since for MC Dropout to work, dropout
        masks should be sampled both during training and testing.
        """
        return super(type(self), self).train(mode=True)

    # def reg(self):
    #     # dropout_regularizer = 2 / (tau * N)
    #     # weight_regularizer = l**2 / (tau * N)
    #     # weight_regularizer / dropout_regularizer = l**2 / 2
    #     # The factor 2 should be ignored for cross-entropy loss,
    #     # and used only for the Euclidean loss.

    #     def get_layer_reg(layer, dropout):
    #         w_reg = torch.stack([
    #             param.norm()**2 for param in layer.parameters()
    #         ]).sum() * self.weight_regularizer / (1 - dropout.drop_prob)

    #         d_reg = dropout.drop_prob * torch.log(dropout.drop_prob) + (
    #             1. - dropout.drop_prob) * torch.log(1. - dropout.drop_prob)
    #         d_reg *= self.dropout_regularizer * x[0].numel()

    #         return d_reg + w_reg

    #     reg = 0
    #     for layer, dropout in layers:
    #         reg += get_layer_reg(layers, dropout)


######################
## Concrete Dropout ##
######################


class ConcreteDropout(nn.Module):
    def __init__(self, p_logit=-2.0, temp=0.01, eps=1e-8):
        super(ConcreteDropout, self).__init__()
        self.p_logit = nn.Parameter(torch.Tensor([p_logit]))
        self.temp = temp
        self.eps = eps

    @property
    def p(self):
        return torch.sigmoid(self.p_logit)

    def forward(self, x):
        # if self.train():
        unif_noise = torch.rand_like(x)
        drop_prob = torch.log(self.p + self.eps) -\
            torch.log(1 - self.p + self.eps) + \
            torch.log(unif_noise + self.eps) - \
            torch.log(1 - unif_noise + self.eps)

        drop_prob = torch.sigmoid(drop_prob / self.temp)
        random_tensor = 1. - drop_prob
        retain_prob = 1. - self.p
        x *= random_tensor
        x /= retain_prob

        return x
