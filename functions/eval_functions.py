import numpy as np
import matplotlib.pyplot as plt
from functions.core_functions import mean_autocorrelation


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
