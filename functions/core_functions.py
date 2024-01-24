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


def cut_constant_idx(
    series: np.ndarray,
    n_sample: np.ndarray,
    offset: int = 0,
) -> tuple[list[int], np.ndarray]:
    """cut a series such that the subsamples have a constant number of events.
    it is assumed that the magnitudes are ordered as desired (e.g. in time or
    in depth)

    Args:
        series:     array of values
        n_sample:   number of subsamples to cut the series into
        offset:     offset where to start cutting the series

    Returns:
        idx:            indices of the subsamples
        subsamples:     list of subsamples
    """
    n_b = np.round(len(series) / n_sample).astype(int)
    idx = np.arange(offset, len(series), n_b)
    idx = idx[1:]

    subsamples = np.array_split(series[offset:], idx - offset)
    return idx, subsamples


def cut_random_idx(
    series: np.ndarray,
    n_sample: int,
) -> tuple[list[int], np.ndarray]:
    """cut a series at random idx points. it is assumed that the magnitudesa
    are ordered as desired (e.g. in time or in depth)

    Args:
        series:     array of values
        n_sample:   number of subsamples to cut the series into

    Returns:
        idx:            indices of the subsamples
        subsamples:     list of subsamples
    """
    # generate random index
    idx = random.sample(list(np.arange(1, len(series))), n_sample - 1)
    idx = np.sort(idx)

    subsamples = np.array_split(series, idx)
    return idx, subsamples


def cut_random(
    series: np.ndarray,
    n_sample: int,
    order: np.ndarray,
) -> tuple[list[int], np.ndarray]:
    """cut a series at random times.

    Args:
        series:     array of values
        n_sample:   number of subsamples to cut the series into
        order:      array of values that can be used to sort the series.
                this could be e.g. the time or the depth of the events.
                Important: order itself is expected to be sorted.

    Returns:
        idx:            indices of the subsamples
        subsamples:     list of subsamples
        subsample_times:list of times corresponding to the subsamples

    """
    if order is None:
        # value errpor
        raise ValueError("order cannot be None")

    # generate random index
    random_choice = (
        np.random.rand(n_sample - 1) * (order[-1] - order[0]) + order[0]
    )
    idx = np.searchsorted(order, random_choice)
    idx = np.sort(idx)

    subsamples = np.array_split(series, idx)
    return idx, subsamples


def b_samples_pos(
    magnitudes: np.ndarray,
    times: np.ndarray[dt.datetime],
    n_sample: int,
    delta_m: float = 0.1,
    return_idx: bool = False,
    cutting: str = "random_idx",
    order: None | np.ndarray = None,
    offset: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """cut the magnitudes randomly into n_series subsamples and estimate
    b-values

    Args:
        magnitudes:     array of magnitudes
        times:          array of times that correspond to magnitudes
        n_sample:       number of subsamples to cut the data into
        delta_m:        magnitude bin width
        return_idx:     if True, return the indices of the subsamples
        cutting:        method of cutting the data into subsamples. either
                    'random_idx' or 'constant_idx' or 'random'
        order:          array of values that can be used to sort the
                    magnitudes, if left as None, it will be assumed that the
                    desired order is in time.

    """

    if order is None:
        order = times

    # cut
    if cutting == "random_idx":
        idx, mags_chunks = cut_random_idx(magnitudes, n_sample)
    elif cutting == "constant_idx":
        idx, mags_chunks = cut_constant_idx(
            magnitudes, n_sample, offset=offset
        )
    elif cutting == "random":
        idx, mags_chunks = cut_random(magnitudes, n_sample, order)
    else:
        raise ValueError(
            "cutting method not recognized, use either 'random_idx' or "
            "'constant_idx' or 'random' for the cutting variable"
        )
    # cut time in the same way (later for b-positive)
    times_chunks = np.array_split(times, idx)

    # estimate b-values
    b_series = np.zeros(n_sample)
    n_bs = np.zeros(n_sample)

    for ii, mags_loop in enumerate(mags_chunks):
        # sort the magnitudes by their time (only if magnitudes were not
        # ordered by time)
        if order is not None:
            times_loop = times_chunks[ii]
            idx_sorted = np.argsort(times_loop)
            mags_loop = mags_loop[idx_sorted]

        if len(mags_loop) > 2:
            b_series[ii], n_bs[ii] = estimate_b_positive(
                np.array(mags_loop), delta_m=delta_m, return_n=True
            )

    if return_idx is True:
        # return the first index of each subsample
        return b_series, n_bs.astype(int), np.append(0, idx)

    return b_series, n_bs.astype(int)


def b_samples(
    magnitudes: np.ndarray,
    n_sample: int,
    mc: float,
    delta_m: float = 0.1,
    return_idx: bool = False,
    cutting: str = "random_idx",
    order: None | np.ndarray = None,
    offset: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """cut the magnitudes randomly into n_series subsamples and estimate
    b-values

    Args:
        magnitudes:     array of magnitudes
        n_sample:       number of subsamples to cut the data into
        mc:             completeness magnitude
        delta_m:        magnitude bin width, assumed to be 0.1 if not given
        return_idx:     if True, return the indices of the subsamples
        cutting:        method of cutting the data into subsamples. either
                    'random_idx' or 'constant_idx' or 'random'
        order:          array of values that can be used to sort the
                    magnitudes
        offset:         offset where to start cutting the series (only for
                    cutting = 'constant_idx')
    """

    # cut
    if cutting == "random_idx":
        idx, mags_chunks = cut_random_idx(magnitudes, n_sample)
    elif cutting == "constant_idx":
        idx, mags_chunks = cut_constant_idx(
            magnitudes, n_sample, offset=offset
        )
    elif cutting == "random":
        idx, mags_chunks = cut_random(magnitudes, n_sample, order)
    else:
        raise ValueError(
            "cutting method not recognized, use either 'random_idx' or "
            "'constant_idx' or 'random' for the cutting variable"
        )

    # estimate b-values
    b_series = np.zeros(n_sample)
    n_bs = np.zeros(n_sample)

    for ii, mags_loop in enumerate(mags_chunks):
        if len(mags_loop) > 1:
            b_series[ii] = estimate_b_tinti(
                np.array(mags_loop), mc=mc, delta_m=delta_m
            )
            n_bs[ii] = len(mags_loop)

    if return_idx is True:
        # return the first index of each subsample
        return b_series, n_bs.astype(int), np.append(0, idx)

    return b_series, n_bs.astype(int)


def get_acf_random(
    magnitudes: np.ndarray,
    times: np.ndarray[dt.datetime],
    n_sample: int,
    mc: None | float = None,
    delta_m: float = 0.1,
    nb_min: int = 2,
    n: int = 1000,
    transform: bool = True,
    cutting: str = "random_idx",
    order: None | np.ndarray = None,
    b_method="positive",
):
    """estimates the autocorrelation from randomly sampling the magnitudes

    Args:
        magnitudes:     array of magnitudes (not the differences!)
        times:          array of times that correspond to magnitudes
        n_series:       number of series to cut the data into
        mc:             completeness magnitude, only needed if b_method is
                    'tinti'
        delta_m:        magnitude bin width, assumed to be 0.1 if not given
        nb_min:         minimum number of events in a series
        n:              number of random samples
        transform:      if True, transform b-values such that they are all
                    comparable regardless of the number of events used
        cutting:        method of cutting the data into subsamples. either
                    'random_idx' or 'random'
        order:          array of values that can be used to sort the
                    magnitudes, if left as None, it will be assumed that the
                    desired order is in time.
        b_method:       method to use for the b-value estimation. either
                    'positive' or 'tinti'.

    Returns:
        acfs:           array of acfs (for each random sample)
        n_series_used:  array of number of b-values used for the
                    crosscorrelation
    """

    # estimate b-value for all data (note: magnitudes might be ordered by
    # space, therefore sort by time is necessary)
    if b_method == "positive":
        idx = np.argsort(times)
        mags_sorted = magnitudes[idx]
        b_all = estimate_b_positive(mags_sorted, delta_m=delta_m)
    elif b_method == "tinti":
        b_all = estimate_b_tinti(magnitudes, mc=mc, delta_m=delta_m)
    else:
        raise ValueError(
            "b_method not recognized, use either 'positive' or 'tinti'"
        )

    # estimate autocorrelation function for random sampples
    acfs = np.zeros(n)
    n_series_used = np.zeros(n)
    if cutting == "constant":
        # for constant window approach, the sindow has to be shifted exactly
        # the number of samples per estimate
        n = int(len(magnitudes) / n_sample)
    for ii in range(n):
        if b_method == "positive":
            b_series, n_bs = b_samples_pos(
                magnitudes,
                times,
                n_sample,
                delta_m=delta_m,
                cutting=cutting,
                order=order,
                offset=ii,
            )
        elif b_method == "tinti":
            # make sure that order is not none when using cutting method
            # 'random'
            if cutting == "random" and order is None:
                order = times
            b_series, n_bs = b_samples(
                magnitudes,
                n_sample,
                mc,
                delta_m=delta_m,
                cutting=cutting,
                order=order,
                offset=ii,
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
        if sum(idx_inf) > 0:
            warnings.warn(
                "inf encountered in b-series, check what is going on"
            )
        idx_min = n_bs < nb_min
        idx = idx_nan | idx_inf | idx_min
        b_series[idx] = np.mean(b_series[~idx])

        # estimate acf
        acfs[ii] = acf_lag_n(b_series, lag=1)
        if np.isnan(acfs[ii]):
            warnings.warn("nan encountered in acf, check what is going on")

        n_series_used[ii] = sum(np.array(~idx))

    return acfs, n_series_used
