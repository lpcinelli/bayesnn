import math

import torch
import torch.nn.functional as F

from .utils import logsumexp_trick, sigmoid, softmax

# Monte Carlo average loss


def mc_loss(pred_list, y, loss_fn, *args, **kwargs):
    loss_list = [loss_fn(pred, y, *args, **kwargs) for pred in pred_list]
    loss_tensor = torch.stack(loss_list, dim=0)
    mc_loss = torch.mean(loss_tensor)
    return mc_loss


# Predictive likelihoods


def predictive_loglik(pred_list, y, loglik_fn, *args, **kwargs):
    loglik_list = [loglik_fn(pred, y, *args, **kwargs) for pred in pred_list]

    # Sum the outputs computed w/ the same set of weights
    loglik_tensor = torch.stack(loglik_list, dim=0)
    num_loglik = torch.sum(torch.ones_like(loglik_tensor))

    # Compute the avg of the likelihoods for each set of sampled weights
    pred_loglik = logsumexp_trick(loglik_tensor) - torch.log(num_loglik)
    return pred_loglik


####################
## Define metrics ##
####################

# Regression metrics


def mse(mu, y):
    mse = F.mse_loss(mu, y)
    return mse


def rmse(mu, y):
    rmse = torch.sqrt(F.mse_loss(mu, y))
    return rmse


# Classification metrics


def softmax_accuracy(logits, y):
    _, pred_class = torch.max(logits, 1)
    correct = pred_class == y
    acc = correct.float().mean()
    return acc


def sigmoid_accuracy(logits, y):
    pred_class = (logits >= 0).to(torch.long).squeeze()
    correct = pred_class == y
    acc = correct.float().mean()
    return acc


########################
## Define likelihoods ##
########################

# Gaussian


def loglik_gaussian(mu, y, tau):
    logliks = 0.5 * (-math.log(2 * math.pi) + math.log(tau) - tau * (y - mu) ** 2)
    loglik = torch.sum(logliks)
    return loglik


def avneg_loglik_gaussian(mu, y, tau):
    logliks = 0.5 * (-math.log(2 * math.pi) + math.log(tau) - tau * (y - mu) ** 2)
    avneg_loglik = -torch.mean(logliks)
    return avneg_loglik


def stoch_loglik_gaussian(mu, y, tau, train_set_size):
    loglik = -train_set_size * avneg_loglik_gaussian(mu, y, tau)
    return loglik


# Bernoulli


def loglik_bernoulli(logits, y):
    loglik = -F.binary_cross_entropy_with_logits(logits, y.float(), reduction="sum")
    return loglik


def avneg_loglik_bernoulli(logits, y):
    logloss = F.binary_cross_entropy_with_logits(logits, y.float())
    return logloss


def stoch_loglik_bernoulli(logits, y, train_set_size):
    loglik = -train_set_size * avneg_loglik_bernoulli(logits, y)
    return loglik


# Categorical / Multinoulli


def loglik_categorical(logits, y):
    loglik = -F.cross_entropy(logits, y, reduction="sum")
    return loglik


def avneg_loglik_categorical(logits, y):
    logloss = F.cross_entropy(logits, y)
    return logloss


def stoch_loglik_categorical(logits, y, train_set_size):
    loglik = -train_set_size * avneg_loglik_categorical(logits, y)
    return loglik


##################
## Define ELBOs ##
##################


# Gaussian


def elbo_gaussian(mu_list, y, tau, train_set_size, kl):
    mc_loglik = mc_loss(
        pred_list=mu_list,
        y=y,
        loss_fn=stoch_loglik_gaussian,
        tau=tau,
        train_set_size=train_set_size,
    )
    elbo = mc_loglik - kl
    return elbo


def avneg_elbo_gaussian(mu_list, y, tau, train_set_size, kl):
    avneg_elbo = -elbo_gaussian(mu_list, y, tau, train_set_size, kl) / train_set_size
    return avneg_elbo


# Bernoulli


def elbo_bernoulli(logits_list, y, train_set_size, kl):
    mc_loglik = mc_loss(
        pred_list=logits_list,
        y=y,
        loss_fn=stoch_loglik_bernoulli,
        train_set_size=train_set_size,
    )
    elbo = mc_loglik - kl
    return elbo


def avneg_elbo_bernoulli(logits_list, y, train_set_size, kl):
    avneg_elbo = -elbo_bernoulli(logits_list, y, train_set_size, kl) / train_set_size
    return avneg_elbo


# Categorical / Multinoulli


def elbo_categorical(logits_list, y, train_set_size, kl):
    mc_loglik = mc_loss(
        pred_list=logits_list,
        y=y,
        loss_fn=stoch_loglik_categorical,
        train_set_size=train_set_size,
    )
    elbo = mc_loglik - kl
    return elbo


def avneg_elbo_categorical(logits_list, y, train_set_size, kl):
    avneg_elbo = -elbo_categorical(logits_list, y, train_set_size, kl) / train_set_size
    return avneg_elbo


###############################
## Define predictive metrics ##
###############################

# Regression metrics


def predictive_mse(mu_list, y):
    mu_tensor = torch.stack(mu_list, dim=1)
    mu = torch.mean(mu_tensor, dim=1)
    mse = F.mse_loss(mu, y)
    return mse


def predictive_rmse(mu_list, y):
    rmse = torch.sqrt(predictive_mse(mu_list, y))
    return rmse


# Classification metrics


def softmax_predictive_accuracy(logits_list, y):
    probs_list = [softmax(logits, dim=1) for logits in logits_list]
    probs_tensor = torch.stack(probs_list, dim=2)
    probs = torch.mean(probs_tensor, dim=2)
    _, pred_class = torch.max(probs, 1)
    correct = pred_class == y
    pred_acc = correct.float().mean()
    return pred_acc


def sigmoid_predictive_accuracy(logits_list, y):
    probs_list = [sigmoid(logits) for logits in logits_list]
    probs_tensor = torch.stack(probs_list, dim=1)
    probs = torch.mean(probs_tensor, dim=1)
    pred_class = torch.round(probs).to(torch.long).squeeze()
    correct = pred_class == y
    pred_acc = correct.float().mean()
    return pred_acc


###################################
## Define predictive likelihoods ##
###################################

# Gaussian


def predictive_loglik_gaussian(mu_list, y, tau):
    pred_loglik = predictive_loglik(
        pred_list=mu_list, y=y, loglik_fn=loglik_gaussian, tau=tau
    )
    return pred_loglik


def predictive_avneg_loglik_gaussian(mu_list, y, tau):
    pred_avneg_ll = -(1 / len(y)) * predictive_loglik_gaussian(mu_list, y, tau)
    return pred_avneg_ll


# Bernoulli


def predictive_loglik_bernoulli(logits_list, y):
    pred_loglik = predictive_loglik(
        pred_list=logits_list, y=y, loglik_fn=loglik_bernoulli
    )
    return pred_loglik


def predictive_avneg_loglik_bernoulli(logits_list, y):
    pred_avneg_ll = -(1 / len(y)) * predictive_loglik_bernoulli(logits_list, y)
    return pred_avneg_ll


# Categorical / Multinoulli


def predictive_loglik_categorical(logits_list, y):
    pred_loglik = predictive_loglik(
        pred_list=logits_list, y=y, loglik_fn=loglik_categorical
    )
    return pred_loglik


def predictive_avneg_loglik_categorical(logits_list, y):
    pred_avneg_ll = -(1 / len(y)) * predictive_loglik_categorical(logits_list, y)
    return pred_avneg_ll


###########################
## Define KL divergences ##
###########################

# Gaussian


def kl_gaussian(q_mu, q_sigma, p_mu, p_sigma):
    """Computes the analytical KL divergence between the gaussian distributions
    q and p: KL(q||p)

    Arguments:
        q_mu {[type]} -- Mean of (variational posterior) gaussian distribution q
        q_sigma {[type]} -- Std of (variational posterior) gaussian distribution q
        p_mu {[type]} -- Mean of (prior) gaussian distribution q
        p_sigma {[type]} -- Mean of (prior) gaussian distribution q

    Returns:
        [type] -- Exact KL divergence
    """

    var_ratio = (q_sigma / p_sigma).pow(2)
    t1 = ((q_mu - p_mu) / p_sigma).pow(2)
    return 0.5 * torch.sum((var_ratio + t1 - 1 - var_ratio.log()))


def stoch_kl_gaussian(samples, q_mu, q_sigma, p_mu, p_sigma):
    """Computes an approximation to the KL divergence between the gaussian
    distributions q and p: KL(q||p)

    Arguments:
        samples {[type]} -- Samples used to compute the log likelihoods
        q_mu {[type]} -- Mean of (variational posterior) gaussian distribution q
        q_sigma {[type]} -- Std of (variational posterior) gaussian distribution q
        p_mu {[type]} -- Mean of (prior) gaussian distribution q
        p_sigma {[type]} -- Mean of (prior) gaussian distribution q

    Returns:
        [type] -- Approximated KL divergence
    """
    # Compute log likelihood of samples on posterior distribution q
    log_qw = loglik_gaussian(q_mu, samples, 1 / q_sigma)

    # Compute log likelihood of samples on prior distribution p
    log_pw = loglik_gaussian(p_mu, samples, 1 / p_sigma)

    return log_qw - log_pw


# MSE (equivalent to lolik_gaussian under the correct choice)
# Should the MSE loss for the MC Dropout be implemented like this:


def mc_mse(mu_list, y):
    loss = mc_loss(pred_list=mu_list, y=y, loss_fn=mse)
    return loss
