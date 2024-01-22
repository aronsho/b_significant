# global imports
import numpy as np
import datetime as dt
import random
import warnings
from seismostats.analysis.estimate_beta import (
    estimate_b_positive,
    estimate_b_tinti,
)

# local imports
from functions.general_functions import transform_n, acf_lag_n


def cut_constant(
    series: np.ndarray,
    n_sample: np.ndarray,
    times: None | np.ndarray = None,
    offset: int = 0,
) -> tuple[list[int], np.ndarray] | tuple[list[int], np.ndarray, np.ndarray]:
    """cut a series at constant intervals

    Args:
        series:     array of values
        n_sample:   number of subsamples to cut the series into
        times:      array of times corresponding to the values of the series,
                optional
        offset:     offset where to start cutting the series

    Returns:
        idx:            indices of the subsamples
        subsamples:     list of subsamples
        subsample_times:list times corresponding to the subsamples

    """
    n_b = np.round(len(series) / n_sample).astype(int)
    idx = np.arange(offset, len(series), n_b)
    idx = idx[1:]

    subsamples = np.array_split(series[offset:], idx - offset)
    if times is not None:
        subsamples_times = np.array_split(times[offset:], idx - offset)
        return idx, subsamples, subsamples_times

    return idx, subsamples


def cut_random(
    series: np.ndarray, n_sample: int, times: None | np.ndarray = None
) -> tuple[list[int], np.ndarray] | tuple[list[int], np.ndarray, np.ndarray]:
    """cut a series at random points

    Args:
        series:     array of values
        n_sample:   number of subsamples to cut the series into
        times:      array of times corresponding to the values of the series

    Returns:
        idx:            indices of the subsamples
        subsamples:     list of subsamples
        subsample_times:list of times corresponding to the subsamples

    """
    # generate random index
    idx = random.sample(np.arange(1, len(series)), n_sample - 1)

    idx = np.sort(idx)
    subsamples = np.array_split(series, idx)

    if times is not None:
        subsamples_times = np.array_split(times, idx)
        return idx, subsamples, subsamples_times

    return idx, subsamples


def random_samples_pos(
    magnitudes: np.ndarray,
    times: np.ndarray[dt.datetime],
    n_sample: int,
    delta_m: float = 0.1,
    return_idx: bool = False,
):
    """cut the magnitudes randomly into n_series subsamples and estimate
    b-values

    Args:
        magnitudes:     array of magnitudes
        times:          array of times that correspond to magnitudes
        n_sample:       number of subsamples to cut the data into
        delta_m:        magnitude bin width
        return_idx:     if True, return the indices of the subsamples
        cutting:        method of cutting the data into subsamples. either
                    'random' or 'constant'

    """
    # cut
    idx, mags_chunks, times_chunks = cut_random(magnitudes, n_sample, times)

    # estimate b-values
    b_series = np.zeros(n_sample)
    n_bs = np.zeros(len(idx) - 1)

    for ii in range(len(idx) - 1):
        mags_loop = mags_chunks[ii]
        times_loop = times_chunks[ii]

        # sort the magnitudes by their time
        idx_sorted = np.argsort(times_loop)
        mags_loop = mags_loop[idx_sorted]

        b_series[ii], n_bs[ii] = estimate_b_positive(
            np.array(mags_loop), delta_m=delta_m, n_b=True
        )

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
        magnitudes:     array of magnitudes (not the differences!)
        times:          array of times that correspond to magnitudes
        n_series:       number of series to cut the data into
        delta_m:        magnitude bin width
        nb_min:         minimum number of events in a series
        n:              number of random samples
        transform:      if True, transform b-values such that they are all
                    comparable regardless of the number of events used

    Returns:
        acfs:           array of acfs (for each random sample)
        n_series_used:  array of number of b-values used for the
                    crosscorrelation
    """

    # estimate b-value for all data (note: magnitudes might be ordered by
    # space, therefore sort by time is necessary)
    idx = np.argsort(times)
    mags_sorted = magnitudes[idx]
    b_all = estimate_b_positive(mags_sorted, delta_m=delta_m)

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
        if sum(idx_nan) > 0:
            warnings.warn(
                "nan encountered in b-series, check what is going on"
            )
        idx_inf = np.isinf(b_series)
        idx_min = n_bs < nb_min
        idx = idx_nan | idx_inf | idx_min
        b_series[idx] = np.mean(b_series[~idx])

        # estimate acf
        acfs[ii] = acf_lag_n(b_series, lag=1)
        if np.isnan(acfs[ii]):
            warnings.warn("nan encountered in acf, check what is going on")
            acfs[ii] = 0

        n_series_used[ii] = sum(np.array(~idx))

    return acfs, n_series_used


def random_samples(
    magnitudes: np.ndarray,
    n_sample: int,
    mc: float = 0,
    delta_m: float = 0.1,
    return_idx: bool = False,
):
    """cut the magnitudes randomly into n_series subsamples and estimate
    b-values"""
    # cut
    idx, mags_chunks = cut_random(magnitudes, n_sample)

    # estimate b-values
    b_series = np.zeros(n_sample)
    n_bs = np.zeros(len(idx) - 1)

    for ii in range(len(idx) - 1):
        mags_loop = mags_chunks[ii]
        b_series[ii] = estimate_b_tinti(
            np.array(mags_loop), mc=mc, delta_m=delta_m
        )
        n_bs[ii] = len(mags_loop)

    if return_idx is True:
        return b_series, n_bs.astype(int), idx

    return b_series, n_bs.astype(int)


def get_acf_random(
    magnitudes: np.ndarray,
    n_series: int,
    mc: float = 0,
    delta_m: float = 0.1,
    nb_min: int = 2,
    n: int = 1000,
    transform: bool = True,
):
    """estimates the autocorrelation from randomly sampling the magnitudes

    Args:
        magnitudes: array of magnitudes (not the differences!)
        n_series:   number of series to cut the data into
        mc:         completeness magnitude
        delta_m:    magnitude bin width
        nb_min:     minimum number of events in a series
        n:          number of random samples
        transform:  if True, transform b-values such that they are all
                comparable regardless of the number of events used

    Returns:
        acfs (np.array):            array of acfs (for each random sample)
        n_series_used (np.array):   array of number of b-values used for the
                                crosscorrelation
    """

    # estimate b-value for all data
    b_all = estimate_b_tinti(magnitudes, mc=mc, delta_m=delta_m)

    # estimate autocorrelation function for random sampples
    acfs = np.zeros(n)
    n_series_used = np.zeros(n)
    for ii in range(n):
        b_series, n_bs = random_samples_pos(
            magnitudes, n_series, mc=mc, delta_m=delta_m
        )

        # transform b-value
        if transform is True:
            for jj in range(len(b_series)):
                b_series[jj] = transform_n(
                    b_series[jj], b_all, n_bs[jj], np.max(n_bs)
                )

        # filter out nan and inf from b-values
        idx_nan = np.isnan(b_series)
        if sum(idx_nan) > 0:
            warnings.warn(
                "nan encountered in b-series, check what is going on"
            )
        idx_inf = np.isinf(b_series)
        idx_min = n_bs < nb_min
        idx = idx_nan | idx_inf | idx_min
        b_series[idx] = np.mean(b_series[~idx])

        # estimate acf
        acfs[ii] = acf_lag_n(b_series, lag=1)
        if np.isnan(acfs[ii]):
            warnings.warn("nan encountered in acf, check what is going on")
            acfs[ii] = 0

        n_series_used[ii] = sum(np.array(~idx))

    return acfs, n_series_used
