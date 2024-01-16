# global imports
import numpy as np
import datetime as dt
import random
from seismostats import estimate_b_elst

# local imports
from functions.b_value_functions import transform_n

def acf_lag_n(b_series: np.ndarray, lag: int = 1):
    """calculates the autocorrelation function for a given lag
    Args:
        b_series (np.array): array of b-values
        lag (int): lag for which the acf is calculated

    Returns:
        acf (float): autocorrelation value for the given lag
    """
    if lag == 0:
        acf = 1
    else:
        acf = sum(
            (b_series[lag:] - np.mean(b_series))
            * (b_series[:-lag] - np.mean(b_series))
        )
        # print(sum((b_series - np.mean(b_series)) ** 2), "sum")
        # print(len(b_series), np.mean(b_series), "len, mean")
        acf /= sum((b_series - np.mean(b_series)) ** 2)
    return acf


def random_samples_pos(
    magnitudes: np.ndarray,
    times: np.ndarray[dt.datetime],
    n_series: int,
    delta_m: float = 0.1,
    return_idx: bool = False,
):
    """cut the magnitudes randomly into n_series subsamples and estimate
    b-values"""
    # generate random index

    idx = random.sample(np.arange(len(magnitudes)) + 1, n_series)
    # idx = np.argsort(np.random.random(len(magnitudes)))[:n_series]
    idx = np.sort(idx)

    # estimate b-values
    b_series = np.zeros(n_series)
    n_bs = np.zeros(len(idx) - 1)

    mags_chunks = np.array_split(magnitudes, idx)
    times_chunks = np.array_split(times, idx)

    for ii in range(len(idx) - 1):
        mags_loop = magnitudes[idx[ii] : idx[ii + 1]]
        times_loop = times[idx[ii] : idx[ii + 1]]

        # sort the magnitudes by their time
        idx_sorted = np.argsort(times_loop)
        mags_loop = mags_loop[idx_sorted]

        b_series[ii] = estimate_b_elst(np.array(mags_loop), delta_m=delta_m)

        # number of events in each subsample (after taking the difference)
        mags_loop = np.diff(mags_loop)
        mags_loop = mags_loop[mags_loop >= delta_m]
        n_bs[ii] = len(mags_loop)

    if return_idx is True:
        return b_series, n_bs.astype(int), idx

    return b_series, n_bs.astype(int)


def get_acf_random_pos(
    magnitudes: np.ndarray,
    times: np.ndarray[dt.datetime],
    n_series: int,
    delta_m: float = 0.1,
    nb_min: int = 2,
    n: int = 1000,
    transform: bool = True,
):
    """estimates the autocorrelation from randomly sampling the magnitudes

    Args:
        magnitudes (np.array): array of magnitudes (not the differences!)
        times (np.array): array of times that correspond to magnitudes
        n_series (int): number of series to cut the data into
        delta_m (float): magnitude bin width
        nb_min (int): minimum number of events in a series
        n (int): number of random samples
        transform (bool): if True, transform b-values such that they are all
            comparable regardless of the number of events used

    Returns:
        acfs (np.array):            array of acfs (for each random sample)
        n_series_used (np.array):   array of number of b-values used for the
                                    crosscorrelation
    """

    # estimate b-value for all data (note: magnitudes might be ordered by space,
    # therefore sort by time is necessary)
    idx = np.argsort(times)
    mags_sorted = magnitudes[idx]
    b_all = estimate_b_elst(mags_sorted, delta_m=delta_m)

    # estimate autocorrelation function for random sampples
    acfs = np.zeros(n)
    n_series_used = np.zeros(n)
    for ii in range(n):
        b_series, n_bs = random_samples_pos(
            magnitudes, times, n_series, delta_m=delta_m
        )

        # transform b-value
        if transform is True:
            for jj in range(len(b_series)):
                b_series[jj] = transform_n(
                    b_series[jj], b_all, n_bs[jj], np.max(n_bs)
                )

        # filter out nan and inf from b-values
        idx_nan = np.isnan(b_series)
        idx_inf = np.isinf(b_series)
        # filter out results that use less datapoints than nb_min
        idx_min = n_bs < nb_min
        idx = idx_nan | idx_inf | idx_min
        b_series[idx] = np.mean(b_series[~idx])

        # estimate acf
        acfs[ii] = acf_lag_n(b_series, lag=1)
        if np.isnan(acfs[ii]):
            print(b_series, "nan encountered in acf, check what is going on")

        n_series_used[ii] = sum(np.array(idx_min))

    return acfs, n_series_used