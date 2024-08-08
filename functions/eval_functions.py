import numpy as np
from scipy.stats import norm


def mu_sigma_mac(
    n_series: np.ndarray,
    cutting: str = "constant_idx",
    gamma: float | None = None
) -> np.ndarray:
    if gamma is None:
        gamma = gamma_factor(cutting)
    mu = -1 / n_series
    sigma = gamma * (n_series - 2) / (n_series * np.sqrt(n_series - 1))
    return mu, sigma


def pval_mac(
    mac: np.ndarray,
    n_series: np.ndarray,
    cutting="constant_idx",
) -> np.ndarray:
    """estimate the p-value of the hyptothesis that the b-value is constant. A
    small p-value indicates that the mean autocorrelation is significantly
    larger than expected by chance, therefore disvalidating the nullhyptothesis

    Args:
        mac:        mean autocorrelation
        n_series:   number of series used for the mean autocorrelation
        cutting:    method of cutting the data into subsamples. either
                'random_idx' or 'constant_idx' or 'random'

    Returns:
        p:          p-value of the hypothesis that the b-value is constant
        z:          z-value
    """
    mu, sigma = mu_sigma_mac(n_series, cutting)
    p = 1 - norm(loc=mu, scale=sigma).cdf(mac)
    return p


def zval_mac(
    mac: np.ndarray,
    n_series: np.ndarray,
    cutting="constant_idx",
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """estimate the z-value of the hyptothesis that the b-value is constant. A
    small p-value indicates that the mean autocorrelation is significantly
    larger than expected by chance, therefore disvalidating the nullhyptothesis

    Args:
        mac:        mean autocorrelation
        n_series:   number of series used for the mean autocorrelation
        cutting:    method of cutting the data into subsamples. either
                'random_idx' or 'constant_idx' or 'random'
        return_z:   if True, return the z-value

    Returns:
        z:          z-value
    """
    mu, sigma = mu_sigma_mac(n_series, cutting)
    z = (mac - mu) / sigma
    return z


def gamma_factor(cutting):
    """return the gamma factor for the given cutting method"""
    if cutting == "constant_idx":
        gamma = 0.81  # n_sample_min = 15
    elif cutting == "random_idx":
        gamma = 0.46  # n_sample_min = 50
    elif cutting == "random":
        gamma = 0.45  # n_sample_min = 50
    else:
        raise ValueError(
            "cutting method not recognized, use either 'random_idx' or "
            "'constant_idx' or 'random' for the cutting variable"
        )
    return gamma
