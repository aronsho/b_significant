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
    n_series_list: list[int] | None = None,
    n_bs: list[int] | None = None,
    cutting: str = "random_idx",
    transform: bool = True,
    b_method: str = "positive",
    plotting: bool = True,
    ax: None | plt.Axes = None,
    parameter: str | None = None,
) -> tuple[float, float, float]:
    # just in case the mc was not filtered out
    idx = mags > mc
    mags = mags[idx]
    time = time[idx]
    n = len(mags)

    # estimtate the mean autocorrelation for different number of series
    if n_series_list is None:
        if n_bs is None:
            raise ValueError("n_series_list or n_bs must be given")
        else:
            n_series_list = np.round(n / n_bs).astype(int)
            n_series_list_s, _ = np.unique(n_series_list, return_index=True)
            n_series_list = n_series_list[n_series_list >= 20]

    acf_mean = np.zeros(len(n_series_list))
    acf_std = np.zeros(len(n_series_list))
    n_series_used = np.zeros(len(n_series_list))
    for ii, n_sample in enumerate(n_series_list):
        acf_mean[ii], acf_std[ii], n_series_used[ii] = mean_autocorrelation(
            mags,
            time,
            n_sample=n_sample,
            mc=mc,
            delta_m=delta_m,
            n=500,
            transform=transform,
            cutting=cutting,
            b_method=b_method,
        )

    # plot the mean autocorrelation
    if plotting:
        if ax is None:
            ax = plt.subplots(figsize=(10, 6))[1]

        # Plot 0.05 threshold
        x = np.arange(min(n_series_list), max(n_series_list) + 1, 0.1)
        mu, sigma = mu_sigma_mac(x, cutting)
        ax.plot(
            n / x,
            1.96 * sigma - 1 / x,
            color="grey",
            linestyle="--",
            alpha=0.5,
        )
        ax.plot(
            n / x,
            -1.96 * sigma - 1 / x,
            color="grey",
            linestyle="--",
            alpha=0.5,
        )
        ax.plot(n / x, mu, color="grey", linestyle="-")
        ax.fill_between(
            n / x,
            1.96 * sigma - 1 / x,
            -1.96 * sigma - 1 / x,
            color="orange",
            alpha=0.1,
            label="95% confidence interval",
        )

        plt.errorbar(
            n / n_series_list,
            acf_mean,
            yerr=1.96 * acf_std,
            marker="o",
            markersize=2,
            color="blue",
            linestyle="none",
            ecolor="lightblue",
            label="Data",
        )
        plt.xlabel("Number of magnitudes per estimate")
        plt.ylabel("Mean autocorrelation")
        plt.legend(loc="lower left")

    if parameter == "z_value":
        z_val = zval_mac(acf_mean, n_series_list, cutting)
        return z_val, acf_std, n_series_used

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
        gamma = 0.8088658668341759  # n_sample_min = 25
    elif cutting == "random_idx":
        gamma = 0.4554834807310068  # n_sample_min = 50
    elif cutting == "random":
        gamma = 0.44930660582990684  # n_sample_min = 50
    else:
        raise ValueError(
            "cutting method not recognized, use either 'random_idx' or "
            "'constant_idx' or 'random' for the cutting variable"
        )
    return gamma
