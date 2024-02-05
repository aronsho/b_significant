import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from functions.core_functions import mean_autocorrelation
import warnings


def MAC_different_n(
    mags: np.ndarray,
    time: np.ndarray,
    mc: float,
    delta_m: float,
    n_series_list: list[int],
    cutting: str = "random_idx",
    transform: bool = True,
    b_method: str = "positive",
    plotting: bool = True,
    ax: None | plt.Axes = None,
) -> tuple[float, float, float]:
    idx = mags > mc
    acf_mean = np.zeros(len(n_series_list))
    acf_std = np.zeros(len(n_series_list))
    n_series_used = np.zeros(len(n_series_list))
    for ii, n_sample in enumerate(n_series_list):
        acf_mean[ii], acf_std[ii], n_series_used[ii] = mean_autocorrelation(
            mags[idx],
            time[idx],
            n_sample=n_sample,
            mc=mc,
            delta_m=delta_m,
            n=500,
            transform=transform,
            cutting=cutting,
            b_method=b_method,
        )

    return acf_mean, acf_std, n_series_used


def mu_sigma_mac(
    n_series: np.ndarray,
    cutting="constant_idx",
) -> np.ndarray:
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
    larger than expected by chance, therefore disvalidating the null-hyptothesis.

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
    return_z=False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """estimate the z-value of the hyptothesis that the b-value is constant. A
    small p-value indicates that the mean autocorrelation is significantly
    larger than expected by chance, therefore disvalidating the null-hyptothesis.

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
        gamma = 0.8088658668341759
    elif cutting == "random_idx":
        warnings.warn(
            "random cutting is not implemented yet, using constant_idx instead"
        )
        gamma = 1
    elif cutting == "random":
        warnings.warn(
            "random cutting is not implemented yet, using constant_idx instead"
        )
        gamma = 1
    else:
        raise ValueError(
            "cutting method not recognized, use either 'random_idx' or "
            "'constant_idx' or 'random' for the cutting variable"
        )
    return gamma
