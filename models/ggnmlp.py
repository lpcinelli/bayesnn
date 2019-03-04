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
        self.act = select_act(act_func)

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
