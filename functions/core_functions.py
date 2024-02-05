# global imports
import numpy as np
import datetime as dt
import random
from scipy.stats import norm
import warnings
from seismostats.analysis.estimate_beta import (
    estimate_b_positive,
    estimate_b_tinti,
)

# local imports
from functions.general_functions import (
    transform_n,
    acf_lag_n,
    utsu_test,
    update_welford,
    finalize_welford,
)


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

    if offset > n_b:
        warnings.warn(
            "offset is larger than the number of events per subsample, this"
            "will lead to cutting off more events than necessary"
        )

    subsamples = np.array_split(series, idx)

    return idx, subsamples


def cut_random_idx(
    series: np.ndarray,
    n_sample: int,
    n_min: int | None = None,
) -> tuple[list[int], np.ndarray]:
    """cut a series at random idx points. it is assumed that the magnitudesa
    are ordered as desired (e.g. in time or in depth)

    Args:
        series:     array of values
        n_sample:   number of subsamples to cut the series into
        n_min:     minimum number of events in a subsample

    Returns:
        idx:            indices of the subsamples
        subsamples:     list of subsamples
    """
    if n_min is None:
        # generate random index
        idx = random.sample(list(np.arange(1, len(series))), n_sample - 1)
        idx = np.sort(idx)
    elif n_min < 4:
        # make sure that there are at least n_min events in each subsample
        # if n_min is very small, most of the time it is faster to trial and
        # error
        check = False
        count = 0
        while check is False and count < 1e5:
            # generate random index
            idx = random.sample(list(np.arange(1, len(series))), n_sample - 1)
            idx = np.sort(idx)
            if (
                min(np.diff(np.concatenate(([0], idx, [len(series)]))))
                >= n_min
            ):
                check = True
            count += 1
        if count == 1e5:
            raise ValueError(
                "could not find a solution for the given parameters. try"
                " making nb_min smaller"
            )
    else:
        # make sure that there are at least n_min events in each subsample
        # if n_min larger, then it is faster to exclude already chosen values
        # and surrounding ones
        idx = np.zeros(n_sample - 1)
        available = list(np.arange(1, len(series)))
        for ii in range(n_sample - 1):
            if len(available) < 1:
                raise ValueError(
                    "could not find a solution for the given parameters. try"
                    " making nb_min smaller"
                )
            idx[ii] = random.sample(available, 1)[0]
            idx_loop = available.index(idx[ii])
            for jj in range(-n_min + 1, n_min):
                if idx_loop - jj >= 0 and idx_loop - jj < len(available):
                    available.pop(idx_loop - jj)
            idx = np.sort(idx).astype(int)

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
    nb_min: int = None,
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
        offset:         offset where to start cutting the series (only for
                    cutting = 'constant_idx')
        nb_min:         minimum number of events in a subsample (only for the
                    cutting method 'random_idx' relevant)

    """

    if order is None:
        order = times

    # cut
    if cutting == "random_idx":
        idx, mags_chunks = cut_random_idx(magnitudes, n_sample, n_min=nb_min)
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

    # make sure that data at the edges is not included if not enough samples
    n_b = np.round(len(magnitudes) / n_sample).astype(int)
    if len(mags_chunks[-1]) < n_b:
        mags_chunks.pop(-1)
        times_chunks.pop(-1)
    else:
        idx = np.concatenate((idx, [len(magnitudes)]))
    if len(mags_chunks[0]) < n_b:
        mags_chunks.pop(0)
        times_chunks.pop(0)
    else:
        idx = np.concatenate(([0], idx))

    # estimate b-values
    b_series = np.zeros(len(mags_chunks))
    n_bs = np.zeros(len(mags_chunks))

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
        else:
            b_series[ii] = np.nan

    if return_idx is True:
        # idx works sch that mags[idx[i]:idx[i+1]] is the i-th subsample. idx
        # has therefore n_sample + 1 elements
        return b_series, n_bs.astype(int), idx

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
    nb_min: int = None,
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
        nb_min:         minimum number of events in a subsample (only for the
                    cutting method 'random_idx' relevant)
    """

    # cut
    if cutting == "random_idx":
        idx, mags_chunks = cut_random_idx(magnitudes, n_sample, n_min=nb_min)
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

    # make sure that data at the edges is not included if not enough samples
    n_b = np.round(len(magnitudes) / n_sample).astype(int)
    if len(mags_chunks[-1]) < n_b:
        mags_chunks.pop(-1)
    else:
        idx = np.concatenate((idx, [len(magnitudes)]))
    if len(mags_chunks[0]) < n_b:
        mags_chunks.pop(0)
    else:
        idx = np.concatenate(([0], idx))

    # estimate b-values
    b_series = np.zeros(len(mags_chunks))
    n_bs = np.zeros(len(mags_chunks))

    for ii, mags_loop in enumerate(mags_chunks):
        if len(mags_loop) > 1:
            b_series[ii] = estimate_b_tinti(
                np.array(mags_loop), mc=mc, delta_m=delta_m
            )
            n_bs[ii] = len(mags_loop)
        else:
            b_series[ii] = np.nan

    if return_idx is True:
        # return the index where the magnitudes where cut- the index is
        # always the first event of the next subsample.
        # Example: The first subsample consists of the magnitudes from (offset)
        # to (idx[0] - 1), and the last subsample consists of the magnitudes
        # from (idx[-1]) to the end of the magnitudes.
        return b_series, n_bs.astype(int), idx

    return b_series, n_bs.astype(int)


def autocorrelation(
    magnitudes: np.ndarray,
    times: np.ndarray[dt.datetime],
    n_sample: int,
    mc: None | float = None,
    delta_m: float = 0.1,
    nb_min: int = 2,
    n: int = 1,
    transform: bool = True,
    cutting: str = "random_idx",
    order: None | np.ndarray = None,
    b_method="positive",
) -> tuple[np.ndarray, np.ndarray]:
    """estimates the autocorrelation from subsampling the magnitudes into
    n_sample pieces, according to the method given in cutting. This process
    will be done n=1 times as default, but can be increased as wished.

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
    if cutting == "constant_idx":
        # for constant window approach, the sindow has to be shifted exactly
        # the number of samples per estimate (minus one for no repititions)
        n = int(len(magnitudes) / n_sample) - 1
    acfs = np.zeros(n)
    n_series_used = np.zeros(n)
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
                nb_min=None,  # it seems that it doesnt make a big difference
                # if this is filtered here or later, therefore it will be done
                # later as more efficient computationally.
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
                nb_min=None,  # it seems that it doesnt make a big difference
                # if this is filtered here or later, therefore it will be done
                # later as more efficient computationally.
            )
        # transform b-value
        if transform is True:
            b_series = transform_n(b_series, b_all, n_bs, np.max(n_bs))

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


def mean_autocorrelation(
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
) -> tuple[float, float, float]:
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
    acfs, n_series_used = autocorrelation(
        magnitudes,
        times,
        n_sample,
        mc,
        delta_m,
        nb_min,
        n,
        transform,
        cutting,
        order,
        b_method,
    )

    mean_acf = np.mean(acfs)
    std_acf = np.std(acfs)
    mean_n_series_used = np.mean(n_series_used)

    return mean_acf, std_acf, mean_n_series_used


def utsu_probabilities(
    magnitudes: np.ndarray,
    times: np.ndarray[dt.datetime],
    n_sample: int,
    mc: None | float = None,
    delta_m: float = 0.1,
    nb_min: int = 2,
    n: int = 1,
    transform: bool = True,
    cutting: str = "constant_idx",
    order: None | np.ndarray = None,
    b_method="positive",
) -> tuple[np.ndarray, np.ndarray]:
    """estimates the p-values that two consecutive values of the b-value
    series are from the same distribution (low p-value means that the
    underlying distributions are probably different)

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
        utsu_p:         array of p-values
        b-series:       array of b-values
        idxs:           indices of where the p-values were estimated (the
                    first idx of the second sample each is used, therefore
                    exaclty where the change actually happend)

    """

    utsu_p = np.zeros(len(magnitudes))
    utsu_p[:] = np.nan
    aggregate_utsu_p = np.zeros((len(magnitudes), 3))

    mean_b = np.zeros(len(magnitudes))
    std_b = np.zeros(len(magnitudes))
    aggregate_b = np.zeros((len(magnitudes), 3))

    if cutting == "constant_idx":
        # for constant window approach, the sindow has to be shifted exactly
        # the number of samples per estimate (minus one for no repititions)
        n = int(len(magnitudes) / n_sample)

    for ii in range(n):
        if b_method == "positive":
            b_series, n_bs, idxs = b_samples_pos(
                magnitudes,
                times,
                n_sample,
                delta_m=delta_m,
                cutting=cutting,
                order=order,
                offset=ii,
                nb_min=None,
                return_idx=True,
            )
        elif b_method == "tinti":
            # make sure that order is not none when using cutting method
            # 'random'
            if cutting == "random" and order is None:
                order = times
            b_series, n_bs, idxs = b_samples(
                magnitudes,
                n_sample,
                mc,
                delta_m=delta_m,
                cutting=cutting,
                order=order,
                offset=ii,
                nb_min=None,
                return_idx=True,
            )

        for idx, b, n_b in zip(idxs[1:], b_series, n_bs[1:]):
            if np.isnan(b) or np.isinf(b) or n_b < nb_min:
                pass
            else:
                aggregate_b[idx - 1] = update_welford(aggregate_b[idx - 1], b)

        for idx, b1, b2, n1, n2 in zip(
            idxs[1:-1], b_series[:-1], b_series[1:], n_bs[:-1], n_bs[1:]
        ):
            if (
                np.isnan(b1)
                or np.isnan(b2)
                or np.isinf(b1)
                or np.isinf(b2)
                or n1 < nb_min
                or n2 < nb_min
            ):
                pass
            else:
                aggregate_utsu_p[idx - 1] = update_welford(
                    aggregate_utsu_p[idx - 1], utsu_test(b1, b2, n1, n2)
                )
                aggregate_b[idx - 1] = update_welford(aggregate_b[idx - 1], b1)

    for ii in range(len(magnitudes)):
        utsu_p[ii], _ = finalize_welford(aggregate_utsu_p[ii])
        mean_b[ii], std_b[ii] = finalize_welford(aggregate_b[ii])
    std_b = np.sqrt(std_b)

    return utsu_p, mean_b, std_b
