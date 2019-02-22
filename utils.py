import torch
import torch.nn.functional as F

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
