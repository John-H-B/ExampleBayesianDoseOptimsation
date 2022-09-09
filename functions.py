import numpy as np
from scipy.stats import beta, norm

from setup import Setup

_s = Setup()


def invlogit(z):
    """
    A linkage functions for logistic regression.
    :param z: A numeric value
    :return: A value between [0,1]
    """
    return 1.0 / (1.0 + np.exp(-z))


def efficacy_model(x, max_eff, mid, scale):
    """
    A three parameter dose-efficacy model
    :param x: The log-transformed dose value.
    :param max_eff: The maximum probability of efficacious response for any possible dose
    :param mid: The inflection point of the dose-efficacy curve
    :param scale: The steepness of the dose-efficacy curve
    :return: A probability of efficacy for a given dose given the parameter values.
    """
    z = scale * (x - mid)
    return max_eff * invlogit(z)


def toxicity_model(x, mid, scale):
    """
    A two parameter dose-toxicity model
    :param x: The log-transformed dose value.
    :param mid: The inflection point of the dose-toxicity curve
    :param scale: The steepness of the dose-toxicity curve
     :return: A probability of toxicity for a given dose given the parameter values.
        """
    z = scale * (x - mid)
    return invlogit(z)


def log_likelihood_eff(theta, x, y,):
    n, s = y
    max_eff, mid, scale = theta
    if max_eff > 1:
        return -np.inf
    pred_p = efficacy_model(x, max_eff, mid, scale)
    pred_not_p = 1 - pred_p
    f = n - s

    LL = 0
    LL += np.log(pred_p) * s
    LL += np.log(pred_not_p) * f
    return np.sum(LL)


max_eff_prior = beta(_s.eff_max_prior_alpha, _s.eff_max_prior_beta)
mid_eff_prior = norm(_s.eff_mid_prior_mu, _s.eff_mid_prior_sigma)


def log_prior_eff(theta):
    max_eff, mid, scale = theta
    if 0 < scale < 10.0:
        return np.log(max_eff_prior.pdf(max_eff)) + np.log(mid_eff_prior.pdf(mid))
    return -np.inf


def log_probability_eff(theta, x, y):
    lp = log_prior_eff(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_eff(theta, x, y)


def log_likelihood_tox(theta, x, y):
    n, s = y
    mid, scale = theta

    if scale < 0:
        return -np.inf
    pred_p = toxicity_model(x, mid, scale)
    pred_not_p = 1 - pred_p
    f = n - s

    LL = 0
    LL += np.log(pred_p) * s
    LL += np.log(pred_not_p) * f
    return np.sum(LL)


mid_tox_prior = norm(_s.tox_mid_prior_mu, _s.tox_mid_prior_sigma)


def log_prior_tox(theta):
    mid_tox, scale_tox = theta
    if scale_tox > 0:
        return np.log(mid_eff_prior.pdf(mid_tox))
    return -np.inf


def log_probability_tox(theta, x, y):
    lp = log_prior_tox(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_tox(theta, x, y)


def utility(p_eff, p_tox):
    """
    A simple utility function that maximises the probability of safe and effective treatment.
    This assumes mutual independence between outcomes.
    :param p_eff:
    :param p_tox:
    :return:
    """
    return p_eff * (1 - p_tox)


pi1 = 0.4
pi2 = 0.17
rho = 3


def utility_efftox(efficacy_probabilities, toxicity_probabilities):
    """
    The efftox binary outcome utility function proposed by Thall et al.
    :param efficacy_probabilities:
    :param toxicity_probabilities:
    :return: A utility value, with utility greater than 0 being a required criteria.
    """
    first_part = (1 - efficacy_probabilities) / (1 - pi1)
    second_part = toxicity_probabilities / pi2
    addition = (first_part ** rho) + (second_part ** rho)
    utility = 1 - (addition ** (1 / rho))
    return utility
