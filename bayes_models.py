import numpy as np
from scipy import stats

def calc_gamma_params(mu, var):
    """
    Calcula alpha y beta para una distribución Gamma dada su media y varianza.
    Asume la parametrización rate (beta = tasa).
    E[lambda] = alpha / beta
    Var(lambda) = alpha / beta^2
    """
    beta = mu / var
    alpha = mu * beta
    return alpha, beta

def gamma_poisson_posterior(alpha_prior, beta_prior, y):
    """
    Calcula los parámetros de la posterior Gamma tras observar un dato Poisson y.
    """
    alpha_post = alpha_prior + y
    beta_post = beta_prior + 1
    return alpha_post, beta_post

def calc_beta_params(mu, var):
    """
    Calcula alpha y beta para una distribución Beta dada su media y varianza.
    """
    # mu = alpha / (alpha + beta)
    # var = (alpha * beta) / ((alpha + beta)^2 * (alpha + beta + 1))
    
    # Resolviendo el sistema algebraico:
    term = (mu * (1 - mu) / var) - 1
    alpha = mu * term
    beta = (1 - mu) * term
    return alpha, beta

def beta_binomial_posterior(alpha_prior, beta_prior, x, n):
    """
    Calcula los parámetros de la posterior Beta tras observar x éxitos en n intentos Binomiales.
    """
    alpha_post = alpha_prior + x
    beta_post = beta_prior + (n - x)
    return alpha_post, beta_post
