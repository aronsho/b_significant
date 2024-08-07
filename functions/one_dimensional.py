# global imports
import numpy as np
import random
import warnings
from seismostats.analysis.estimate_beta import (
    estimate_b_more_positive,
    estimate_b,
)

# local imports
from functions.general_functions import (
    transform_n,
    a_samples,
    b_samples,
)
from functions.eval_functions import mu_sigma_mac


def acf_lag_n(values: np.ndarray, lag: int = 1) -> float:
    """calculates the autocorrelation function of a series for a given lag
    Args:
        values: array of b-values
        lag:    lag for which the acf is calculated

    Returns:
        acf: autocorrelation value for the given lag. Only non-nan
                values are considered
        n:  number of non-nan values used for the calculation
    """
    # estimate mean
    n = len(values[~np.isnan(values)])
    mean_b = np.mean(values[~np.isnan(values)])
    # do not consider nan values
    values[np.isnan(values)] = mean_b

    if lag == 0:
        acf = 1
    else:
        acf = sum(
            (values[lag:] - np.mean(values))
            * (values[:-lag] - np.mean(values))
        )
        acf /= sum((values - np.mean(values)) ** 2)
    return acf, n


def mac_one_dimension(
        order,
        mags,
        delta_m,
        mc,
        times,
        n_points,
        n_realizations,
        eval_coords=None,
        min_num=10,
        b_method="positive",
        partitioning="constant_idx",
        transform=False,
        image_tech='right',
        scaling_factor=1,
        return_nm=False,
        include_a=False,
):
    """
    This function estimates the mean autocorrelation for the one-dimensional
    case (along the dimension of order). Additionally, it provides the mean
    a- and b-values for each grid-point. The partitioning method is based on
    voronoi tesselation (random area).

    Args:
        order:  orderable dimension along which the analysis is performed.
            It should be a one-dimensional array.
        mags:   magnitudes of the events
        delta_m:    magnitude bin width
        mc:     completeness magnitude
        times:  times of the events
        limit:  limits of the ordeable dimension
        n_points:   number of partitions the data is cut into
        n_realizations: number of realizations for the estimation of the mean
        eval_coords:    coordinates at which the a- and b-values are estimated
        min_num:    minimum number of events in a partition
        b_method:   method to estimate the b-values
        partitioning:   method to partition the data. Either 'constant_idx',
            'random_idx' or 'random'
        transform:  if True, the b-values are transformed to be comparable
        image_tech: technique to estimate the image of the voronoi tesselation.
            Either 'right', 'center', or 'average'
        scaling_factor: factor to scale the overall volume. It could for
            example the total length of the orderable dimension (e.g. the
            number of years the data spans)
        return_nm:  if True, the mean number of events per b-value estimate is
            returned
        include_a:  if True, the a-values and the a-value significance are
            returned

    Returns:
        mac:    mean autocorrelation
        mu_mac: expected mean autocorrelation und H0
        std_mac:    expected standard deviation of the mean autocorrelation
                under H0
        a_average:  mean a-values
        b_average:  mean b-values
    """

    # 1. some data checks
    if len(mags) != len(order):
        raise ValueError("The number of magnitudes and coordinates do not "
                         "match")

    if not np.all(np.diff(order) >= 0*np.diff(order)[0]):
        raise ValueError("The order variable is not ordered")

    if min(mags) < mc:
        raise ValueError("The completeness magnitude is larger than the "
                         "smallest magnitude")

    if partitioning == 'random_idx' or partitioning == 'random':
        if eval_coords is None:
            eval_coords = order
        if image_tech == 'right' or image_tech == 'center':
            image_tech = 'average'
            warnings.warn("for random_idx, only the average image technique "
                          "is usable")
    elif partitioning == 'constant_idx':
        eval_coords = order
        n_m = np.round(len(mags) / n_points).astype(int)
        if n_realizations > int(len(mags) / n_points):
            warnings.warn("The number of realizations was too large, leading"
                          "to repititions. Therefore it was reduced.")
            n_realizations = int(len(mags) / n_points) + 1

    # 2. estimate a and b values for n realizations
    b_average = np.zeros(len(eval_coords))
    std_b_average = np.zeros(len(eval_coords))
    average_cnt_b = np.zeros(len(eval_coords))
    ac_1D = np.zeros(n_realizations)
    n = np.zeros(n_realizations)
    n_ms = np.zeros(n_realizations)
    if include_a is True:
        a_average = np.zeros(len(eval_coords))
        average_cnt_a = np.zeros(len(eval_coords))
        ac_1D_a = np.zeros(n_realizations)
        n_a = np.zeros(n_realizations)
    for ii in range(n_realizations):
        # 2.1 partition data with the given method
        if partitioning == "constant_idx":
            idx_left, tile_magnitudes = cut_constant_idx(
                mags, n_points, offset=ii
            )
        elif partitioning == "random_idx":
            idx_left, tile_magnitudes = cut_random_idx(mags, n_points)
        elif partitioning == "random":
            idx_left, tile_magnitudes = cut_random(mags, n_points, order)

        # 2.2 cut time in the same way and define idx such that they can be
        # used lateron
        tile_times = np.array_split(times, idx_left)
        idx_left = np.concat([[0], idx_left])
        idx_right = np.concat([idx_left[1:], [len(mags)-1]])

        # 2.3 make sure that data at the edges is not included if not enough
        # samples
        if partitioning == "constant_idx":
            if len(tile_magnitudes[-1]) < n_m:
                tile_magnitudes.pop(-1)
                tile_times.pop(-1)
                idx_left = idx_left[:-1]
                idx_right = idx_right[:-1]
            if len(tile_magnitudes[0]) < n_m:
                tile_magnitudes.pop(0)
                tile_times.pop(0)
                idx_left = idx_left[1:]
                idx_right = idx_right[1:]

        # 2.4 estimate length of each sample (scaled by the scaling factor)
        volume = np.diff(
            np.concat([order[idx_left], [order[idx_right[-1]]]])) / abs(
                order[-1] - order[0]) * scaling_factor

        # 2.5 estimate a- and b-values
        b_vec,  std_b_vec, n_m_loop = b_samples(
            tile_magnitudes, tile_times, delta_m,
            mc, b_method=b_method, return_std=True)
        b_vec[n_m_loop < min_num] = np.nan

        if include_a is True:
            a_vec = a_samples(tile_magnitudes, tile_times, delta_m,
                              mc, volumes=volume, a_method=b_method)
            a_vec[n_m_loop < min_num] = np.nan

        # 2.6 estimate average events per b-value estimate
        n_ms[ii] = np.mean(n_m_loop[n_m_loop > min_num])

        # 2.7 estimate the b-value at each point
        if image_tech == 'right':
            b_average[idx_right] = b_vec
            std_b_average[idx_right] = std_b_vec
            if include_a is True:
                a_average[idx_right] = a_vec
        if image_tech == 'center':
            idx_center = (np.round(idx_left+idx_right)/2).astype(int)
            b_average[idx_center] = b_vec
            std_b_average[idx_center] = std_b_vec
            if include_a is True:
                a_average[idx_center] = a_vec
        if image_tech == 'average':
            for jj, b_loop in enumerate(b_vec):
                if ~np.isnan(b_loop):
                    b_average[idx_left[jj]:idx_right[jj]] += b_loop
                    std_b_average[idx_left[jj]:idx_right[jj]] += std_b_vec[jj]
                    average_cnt_b[idx_left[jj]:idx_right[jj]] += 1
                if include_a is True:
                    if ~np.isnan(a_vec[jj]):
                        a_average[idx_left[jj]:idx_right[jj]] += a_vec[jj]
                        average_cnt_a[idx_left[jj]:idx_right[jj]] += 1

        # 2.8 transform the b-values
        if transform is True:
            idx_t = np.argsort(times)
            mags_sorted = mags[idx_t]
            if b_method == "more_positive":
                b_all = estimate_b_more_positive(mags_sorted, delta_m=delta_m)
            else:
                b_all = estimate_b(
                    mags_sorted, mc, delta_m=delta_m, method=b_method)
            b_vec = transform_n(b_vec, b_all, n_m_loop, np.max(n_m_loop))

        # 2.9 estimate autocorrelation (not considering nan)
        ac_1D[ii],  n[ii] = acf_lag_n(b_vec)

        if include_a is True:
            ac_1D_a[ii],  n_a[ii] = acf_lag_n(a_vec)

    # 3. estimate the averages  & expected standard deviation of MAC
    if image_tech == 'average':
        b_average = b_average / average_cnt_b
        std_b_average = std_b_average / average_cnt_b
    mac = np.mean(ac_1D)
    mean_n = np.mean(n)
    mu_mac, std_mac = mu_sigma_mac(mean_n, partitioning)

    mean_nm = np.mean(n_ms)

    if include_a is True:
        if image_tech == 'average':
            a_average = a_average / average_cnt_a
        mac_a = np.mean(ac_1D_a)
        mean_n_a = np.mean(n_a)
        mu_mac_a, std_mac_a = mu_sigma_mac(mean_n_a, partitioning)

        if return_nm is True:
            return (mac, mu_mac, std_mac, b_average, std_b_average,
                    mac_a, mu_mac_a, std_mac_a, a_average, mean_nm)
        return (mac, mu_mac, std_mac, b_average,
                mac_a, mu_mac_a, std_mac_a, a_average)

    if return_nm is True:
        return mac, mu_mac, std_mac, b_average, std_b_average, mean_nm
    return mac, mu_mac, std_mac, b_average, std_b_average


def cut_constant_idx(
    series: np.ndarray,
    n_sample: np.ndarray,
    offset: int = 0,
) -> tuple[list[int], list[np.ndarray]]:
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
    n_m = np.round(len(series) / n_sample).astype(int)
    idx = np.arange(offset, len(series), n_m)

    if offset == 0:
        idx = idx[1:]

    if offset > n_m:
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
) -> tuple[list[int],  list[np.ndarray]]:
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
) -> tuple[list[int], list[np.ndarray]]:
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

