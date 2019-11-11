import math

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


############################
## Multi-Layer Perceptron ##
############################


class MLP(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_sizes,
            output_size,
            act_func="relu",
            obs_noise_init=1.0,
    ):
        super(type(self), self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        if output_size is not None:
            self.output_size = output_size
            self.squeeze_output = False
        else:
            self.output_size = 1
            self.squeeze_output = True

        # Set observation noise (tuned as precision parameter of gaussian dist.)
        # self.log_noise = nn.Parameter(
        #     torch.Tensor(self.output_size).fill_(math.log(obs_noise_init)))

        # Set activation function
        self.act = select_act(act_func)

        # Define layers
        if len(hidden_sizes) == 0:
            # Linear model
            self.hidden_layers = []
            self.output_layer = nn.Linear(self.input_size, self.output_size)
        else:
            # Neural network
            self.hidden_layers = nn.ModuleList([
                nn.Linear(in_size, out_size)
                for in_size, out_size in zip([self.input_size] +
                                             hidden_sizes[:-1], hidden_sizes)
            ])
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
