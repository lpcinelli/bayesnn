
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import seaborn as sns

sns.set('paper', 'white', 'colorblind', font_scale=2.2,
        rc={'lines.linewidth': 2,
            'lines.markersize': 10,
            'figure.figsize': (12.0, 10.0),
            'image.interpolation': 'nearest',
            'image.cmap': 'gray',
            'text.usetex' : True,
            }
        )

# Link Functions


def softmax(logits, dim=1):
    return F.softmax(logits, dim=dim)


def sigmoid(logits):
    return F.sigmoid(logits)


# Useful functions


def logsumexp_trick(logvars):
    """Equal to common logsumexp function but more numerically stable. It
    removes the highest real value from the exponent and then adds it back
    after the log.

    Arguments:
        logvars {torch.FloatTensor}

    Returns:
        [torch.FloatTensor]
    """
    m, _ = torch.max(logvars, 0)
    logvar = m + torch.log(torch.sum(torch.exp(logvars - m)))
    return logvar


def add_weight_decay(net, l2_value, skip=()):
    """Specifies to which of the `net` params weight decay is going to be
    applied. As default, no decay is applied to biases (and params not being
    optimized). One may further specify other params to by adding their names
    in the tuple `skip`.

    Arguments:
        net {torch.nn.Module} -- network model
        l2_value {float} -- weight decay scalar value (lambda)

    Keyword Arguments:
        skip {tuple} -- name of params to which no decay is applied (default: {()})

    Returns:
        list -- list of dicts to be fed to torch.optimizer, each dict contains
            the set of params for which a `weight_decay` value is applied
            during optimization
    """
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": l2_value},
    ]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def plot_1d_results(data, data_generator, predictions=None, method=None, nb_std=1):
    '''Plot the data samples from data, the 1d data defined by `data_generator` (mean and std skirt)
     and (if available) the mean and variance of the model's predictions. Skirts correspond to 1 std
     deviation.

     Args:
         data (iterable): 1d data samples [x_train, y_train]
         data_generator (dict): {'mean': func that spans x_mean, 'std': func that spans x_std}.
            Formula: x = x_mean + x_std*N(0,1)
        predictions (tuple): (model input, model predictions, noise variance). Types and shapes:
            input: numpy.array
            predictions: numpy.array (nb_mc_samples, nb_data_samples)
            noise variance: float
        nb_std (int): nb of standard deviations the uncertainty skirts in plot should be
    '''
    x_domain = data[0].squeeze()
    train_x = np.arange(x_domain.min().item(),
                        x_domain.max().item(), 1/100)

    # plot the training data distribution
    plt.plot(train_x, data_generator['mean'](train_x), 'red', label='data mean')
    plt.fill_between(train_x,
                     data_generator['mean'](train_x) - nb_std*data_generator['std'](train_x),
                     data_generator['mean'](train_x) + nb_std*data_generator['std'](train_x),
                     color='orange', alpha=0.8, label='data 1-std')
    plt.plot(x_domain, data[1], 'r.', alpha=0.5, label='train sampl')

    # plot the model distribution
    if predictions is not None:
        if method not in ['bbb', 'vadam', 'mcdropout', 'pbp']:
            raise ValueError('incorrect choice of training method {}'.format(method))
        if method == 'pbp':
            y_means = predictions[1][0]
            y_vars = predictions[1][1]
        else:
            y_means = predictions[1].mean(axis=0)
            y_vars = predictions[1].var(axis=0)

        x = predictions[0].squeeze()

        # TODO: So far implemented models do not have support for heteroskedastic noise so we assume
        # `heteroskedastic_part` to be zero. If this is ever implemented, computation from std noise
        # predictions would be: (noises**2).mean(axis = 0)**0.5

        # Variance
        heteroskedastic_part = np.zeros_like(y_vars)
        homoskedastic_part = predictions[2] if len(predictions) == 3 else 0

        aleatoric = heteroskedastic_part + homoskedastic_part
        epistemic = y_vars

        aleatoric = np.minimum(aleatoric, 10e3)
        epistemic = np.minimum(epistemic, 10e3)

        # Standard Deviation
        total_unc = (aleatoric + epistemic)**0.5
        aleatoric = aleatoric**0.5
        epistemic = epistemic**0.5

        plt.plot(x, y_means, label='model mean')
        plt.fill_between(x,
                         y_means - nb_std*aleatoric,
                         y_means + nb_std*aleatoric,
                         color='g', alpha = 0.4, label='aleatoric')
        plt.fill_between(x,
                         y_means - nb_std*total_unc,
                         y_means - nb_std*aleatoric,
                         color='b', alpha = 0.4, label='epistemic')
        plt.fill_between(x,
                         y_means + nb_std*aleatoric,
                         y_means + nb_std*total_unc,
                         color='b', alpha=0.4)
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.ylim([-3,2])
    plt.legend()