# imports
import numpy as np
import rft1d
import scipy
from scipy.stats import norm

from seismostats import simulate_magnitudes, bin_to_precision
from seismostats.analysis.estimate_beta import (
    estimate_b_positive,
    estimate_b_tinti,
)
from seismostats.analysis.estimate_mc import empirical_cdf
import datetime as dt
import warnings


def update_welford(existing_aggregate: tuple, new_value: float) -> tuple:
    """Update Welford's algorithm for computing a running mean and standard
    deviation

    Args:
        existing_aggregate:     (count, mean, M2) where count is the number
                        of values used up tp that point, mean is the mean and
                        M2 is the sum of the squares of the differences from
                        the mean of the previous step
        new_value:              new value of the series of which the standard
                        deviation and mean is to be calculated

    Returns:
        aggregate:  (count, mean, M2) where count is the number of values used
                        up tp that point, mean is the mean and M2 is the sum of
                        the squares of the differences from the mean of this
                        step
    """
    (count, mean, M2) = existing_aggregate
    count += 1
    delta = new_value - mean
    mean += delta / count
    delta2 = new_value - mean
    M2 += delta * delta2
    return (count, mean, M2)


def finalize_welford(existing_aggregate: tuple) -> tuple[float, float]:
    """Retrieve the mean, variance and sample variance from an aggregate

    Args:
        existing_aggregate:  (count, mean, M2) where count is the number
                        of values used up tp that point, mean is the mean and
                        M2 is the sum of the squares of the differences for
                        the whole series of which the standard deviation and
                        mean is to be calculated

    Returns:
        mean:       mean of the series
        variance:   variance of the series
    """
    (count, mean, M2) = existing_aggregate
    if count < 2:
        if count == 1:
            # raise warinng
            warnings.warn(
                "only one value used, therefore variance is not defined"
            )
            return mean, np.nan
        if count == 0:
            # raise warinng
            warnings.warn("no value used, therefore variance is not defined")
            return np.nan, np.nan
    else:
        (mean, variance) = (
            mean,
            M2 / count,
        )
        return mean, variance


def transform_n(
    x: np.ndarray, b: float, n1: np.ndarray, n2: np.ndarray
) -> np.ndarray:
    """transform b-value to be comparable to other b-values

    Args:
        x (float):  b-value to be transformed
        b (float):  true b-value
        n1 (int):   number of events in the distribution to be transformed
        n2 (int):   number of events to which the distribution is transformed

    Returns:
        x (float):  transformed b-value
    """
    x_transformed = b / (1 - np.sqrt(n1 / n2) * (1 - b / x))
    return x_transformed


def inverse_norm(x: np.ndarray, b: float, n: int) -> np.ndarray:
    """distribution function of the reciprocal gaussian distribution. This is
    the distribution of 1/X where X is normally distributed. It is designed
    specifically to be used as proxy of the distribution of b-value estiamtes.

    Args:
        x:      values for which the distribution function is calculated
            (i.e. estimated b-value)
        b:      true b-value
        n:      number of events in the distribution

    Returns:
        dist:   probability density at x
    """
    dist = (
        1
        / b
        / np.sqrt(2 * np.pi)
        * np.sqrt(n)
        * (b / x) ** 2
        * np.exp(-n / 2 * (1 - b / x) ** 2)
    )
    return dist


class inverse_norm_class(scipy.stats.rv_continuous):
    """distribution function of the reciprocal normal distribution.This can be
    used, for instance to
    - compute the cdf
    - generate random numbers that follow the reciprocal normal distribution

    Args:
        b:      true b-value
        n_b:    number of events in the distribution
    """

    def __init__(self, b, n_b):
        scipy.stats.rv_continuous.__init__(self, a=0.0)
        self.b_val = b
        self.n_b = n_b

    def _pdf(self, x):
        return inverse_norm(x, b=self.b_val, n=self.n_b)


def cdf_inverse_norm(x: np.ndarray, b: float, n_b: int) -> np.ndarray:
    """distribution function of the reciprocal gaussian distribution. This is
    the distribution of 1/X where X is normally distributed. It is designed
    specifically to be used as proxy of the distribution of b-value estiamtes.

    Args:
        x:      values for which the distribution function is calculated
            (i.e. estimated b-value)
        b:      true b-value
        n:      number of events in the distribution

    Returns:
        y:   cdf at x
    """

    x = np.sort(x)
    x = np.unique(x)
    y = np.zeros(len(x))
    inverse_normal_distribution = inverse_norm_class(b=b, n_b=n_b)
    y = inverse_normal_distribution.cdf(x=x)

    return x, y


def ks_test_b_dist(
    sample: np.ndarray,
    mc: float,
    delta_m: float,
    n_b,
    ks_ds: list[float] | None = None,
    n: int = 10000,
    b: float | None = None,
) -> tuple[float, float, list[float]]:
    """
    Perform the Kolmogorov-Smirnov (KS) test for the b-value distribution

    Args:
        sample:     Magnitude sample
        mc:         Completeness magnitude
        delta_m:    Magnitude bin size
        ks_ds:      List to store KS distances, by default None
        n:          Number of number of times the KS distance is calculated for
                estimating the p-value, by default 10000
        beta :      Beta parameter for the Gutenberg-Richter distribution, by
                    default None

    Returns:
        orig_ks_d:  original KS distance
        p_val:      p-value
        ks_ds:      list of KS distances
    """

    if b is None:
        b = np.mean(sample) * (n - 1) / n  # taking account for the bias

    if ks_ds is None:
        ks_ds = []
        n_sample = len(sample)

        # We want to compare the ks distance with the distribution that would
        # result from a perfekt GR law. We assume that binning does not have a
        # large impact on the cdf, which is an ok approximation.The test will
        # stil be valid even if the assumption is not correct, as this is true
        # both for the empirical cdf and the synthetical created from which
        # the p_val is retrieved.
        simulated_b = b_synth(
            n * n_sample, b, n_b, mc, delta_m, b_parameter="b_value"
        )

        for ii in range(n):
            simulated = simulated_b[
                n_sample * ii : n_sample * (ii + 1)  # noqa
            ]
            # here, we assume that binning does not have a large impact on the
            # cdf, which is an ok approximation.
            _, y_th = cdf_inverse_norm(simulated, b, n_b)
            _, y_emp = empirical_cdf(simulated)

            ks_d = np.max(np.abs(y_emp - y_th))
            ks_ds.append(ks_d)

    _, y_th = cdf_inverse_norm(sample, b, n_b)
    _, y_emp = empirical_cdf(sample)

    orig_ks_d = np.max(np.abs(y_emp - y_th))
    p_val = sum(ks_ds >= orig_ks_d) / len(ks_ds)

    return orig_ks_d, p_val, ks_ds


def b_synth(
    n: int,
    b: float,
    n_b: int,
    mc: float = 0,
    delta_m: float = 0.1,
    b_parameter: str = "b_value",
) -> float:
    """create estaimted b-values from a given true b-value

    Args:
        n:              number of estimated beta / b-values to simulate
        b:              true beta / b-value
        n_b:            number of events per beta / b-value estimate
        mc:             completeness magnitude
        delta_m:        magnitude bin width
        b_parameter:    'b_value' or 'beta'

    Returns:
        b_synth:    synthetic beta / b-value
    """

    mags = simulated_magnitudes_binned(
        n * n_b, b, mc, delta_m, b_parameter=b_parameter
    )

    b = np.zeros(n)
    for ii in range(n):
        b[ii] = estimate_b_tinti(
            mags[ii * n_b : (ii + 1) * n_b],  # noqa
            mc,
            delta_m,
            b_parameter=b_parameter,
        )
    return b


def acf_lag_n(series: np.ndarray, lag: int = 1) -> float:
    """calculates the autocorrelation function of a series for a given lag
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
            (series[lag:] - np.mean(series))
            * (series[:-lag] - np.mean(series))
        )
        acf /= sum((series - np.mean(series)) ** 2)
    return acf


def simulated_magnitudes_binned(
    n: int,
    b: float | np.ndarray,
    mc: float,
    delta_m: float,
    mag_max: float = None,
    b_parameter: str = "b_value",
) -> np.ndarray:
    """simulate magnitudes and bin them to a given precision. input 'b' can be
    specified to be beta or the b-value, depending on the 'b_parameter' input

    Args:
        n:              number of magnitudes to simulate
        b:              b-value or beta of the distribution from which
                magnitudes are simulated. If b is np.ndarray, it must have the
                length n. Then each magnitude is simulated from the
                corresponding b-value
        mc:             completeness magnitude
        delta_m:        magnitude bin width
        mag_max:        maximum magnitude
        b_parameter:    'b_value' or 'beta'

    Returns:
        mags:   array of magnitudes
    """
    if b_parameter == "b_value":
        beta = b * np.log(10)
    elif b_parameter == "beta":
        beta = b
    else:
        raise ValueError("b_parameter must be 'b_value' or 'beta'")

    mags = simulate_magnitudes(n, beta, mc - delta_m / 2, mag_max)
    if delta_m > 0:
        mags = bin_to_precision(mags, delta_m)
    return mags


def simulate_rectangular(
    n_total: int,
    n_deviation: int,
    b: float,
    delta_b: float,
    mc: float,
    delta_m: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate binned magnitudes with a step of length N_deviation in the
    b-value

    Args:
        n_total:        total number of magnitudes to simulate
        n_deviation:    number of magnitudes with deviating b-value
        b:              b-value of the background
        delta_b:        deviation of b-value
        mc:             completeness magnitude
        delta_m:        magnitude bin width

    Returns:
        magnitudes: array of magnitudes
        b_true:     array of b-values from which each magnitude was simulated

    """
    n_loop1 = int((n_total - n_deviation) / 2)

    b_true = np.ones(n_total) * b
    b_true[n_loop1 : n_loop1 + n_deviation] = b + delta_b  # noqa

    magnitudes = simulated_magnitudes_binned(n_total, b_true, mc, delta_m)
    return magnitudes, b_true


def simulate_step(
    n_total: int,
    b: float,
    delta_b: float,
    mc: float,
    delta_m: float = 0.01,
    idx_step: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate binned magnitudes with a step at idx in the b-value

    Args:
        n_total:              total number of magnitudes to simulate
        b:              b-value of the background
        delta_b:        deviation of b-value
        mc:             completeness magnitude
        delta_m:        magnitude bin width
        idx_step:       index of the magnitude where the step occurs. if None,
                    the step occurs at the middle of the sequence

    Returns:
        magnitudes: array of magnitudes
        b_true:     array of b-values from which each magnitude was simulated

    """

    if idx_step is None:
        idx_step = int(n_total / 2)

    b_true = np.ones(n_total) * b
    b_true[idx_step:] = b + delta_b

    magnitudes = simulated_magnitudes_binned(n_total, b_true, mc, delta_m)
    return magnitudes, b_true


def simulate_sinus(
    n_total: int,
    n_wavelength: int,
    b: float,
    delta_b: float,
    mc: float,
    delta_m: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate binned magnitudes with an underlying sinusoidal b-value
    distribution

    Args:
        n_total:        total number of magnitudes to simulate
        n_wavelength:   wavelength of the sinusoidal
        b:              b-value of the background
        delta_b:        deviation of b-value
        mc:             completeness magnitude
        delta_m:        magnitude bin width

    Returns:
        magnitudes: array of magnitudes
        b_true:     array of b-values from which each magnitude was simulated

    """
    b_true = (
        b
        + np.sin(np.arange(n_total) / (n_wavelength - 1) * 2 * np.pi) * delta_b
    )

    magnitudes = simulated_magnitudes_binned(n_total, b_true, mc, delta_m)
    return magnitudes, b_true


def simulate_ramp(
    n_total: int,
    b: float,
    delta_b: float,
    mc: float,
    delta_m: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate binned magnitudes with an underlying b-value that rises
    constantly

    Args:
        n_total:              total number of magnitudes to simulate
        b:              b-value of the background
        delta_b:        deviation of b-value
        mc:             completeness magnitude
        delta_m:        magnitude bin width

    Returns:
        magnitudes: array of magnitudes
        b_true:     array of b-values from which each magnitude was simulated

    """
    b_true = b + np.arange(n_total) / n_total * delta_b

    magnitudes = simulated_magnitudes_binned(n_total, b_true, mc, delta_m)
    return magnitudes, b_true


def simulate_randomfield(
    n_total: int,
    kernel_width: float,
    b: float,
    b_std: float,
    mc: float,
    delta_m: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate binned magnitudes where the underlying b-values vary with time
    as a random gaussian process

    Args:
        n_total:              total number of magnitudes to simulate
        b:              b-value of the background
        delta_b:        deviation of b-value
        mc:             completeness magnitude
        delta_m:        magnitude bin width
        idx_step:       index of the magnitude where the step occurs. if None,
                    the step occurs at the middle of the sequence

    Returns:
        magnitudes: array of magnitudes
        b_true:     array of b-values from which each magnitude was simulated

    """
    magnitudes = np.zeros(n_total)
    kernel_width
    b_s = abs(b + rft1d.random.randn1d(1, n_total, kernel_width) * b_std)

    for ii in range(n_total):
        magnitudes[ii] = simulate_magnitudes(
            1, b_s[ii] * np.log(10), mc=mc - delta_m / 2
        ).item()
    return bin_to_precision(magnitudes, delta_m), b_s


def utsu_test(
    b1: np.ndarray, b2: np.ndarray, n1: np.ndarray[int], n2: np.ndarray
) -> np.ndarray:
    """Given two b-value estimates from two magnitude samples, this functions
    gives back the probability that the actual underlying b-values are not
    different. All the input arrays have to have the same length.

    Source: TODO Need to verify that this is used in Utsu 1992 !!!

    Args:
        b1:     b-value estimate of first sample
        b2:     b-value estimate of seconds sample
        N1:     number of magnitudes in first sample
        N2:     number of magnitudes in second sample

    Returns:
        p:      Probability that the underlying b-value of the two samples is
            identical
    """
    delta_AIC = (
        -2 * (n1 + n2) * np.log(n1 + n2)
        + 2 * n1 * np.log(n1 + n2 * b1 / b2)
        + 2 * n2 * np.log(n2 + n1 * b2 / b1)
        - 2
    )
    p = np.exp(-delta_AIC / 2 - 2)
    return p


def b_any_series(
    magnitudes: np.ndarray,
    times: np.ndarray[dt.datetime],
    n_b: int,
    delta_m: float = 0,
    mc: float = None,
    return_std: bool = False,
    overlap: float = 0,
    return_bar: bool = False,
    method: str = "tinti",
):
    """estimates the b-value using a constant number of events (n_times).

    Args:
        magnitudes:         array of magnitudes. Magnitudes should be
                            sorted in the way the series is wanted
                            (time sorting is not assumed)
        times:               array of dates
        n_b:                number of events to use for the estimation
        mc:                 completeness magnitude
        delta_m:            magnitude bin width
        return_std:         if True, return the standard deviation of the
                        b-value
        overlap:            fraction of overlap between the time windows
        return_bar:     if True, return the time window lengths
        method:             method to use for the b-value estimation.

    Returns:
        b_time:     array of b-values
        time_max:   array of end times of the time windows
        time_bar:   array of time window lengths
    """
    n_eval = len(magnitudes) - n_b
    b_any = []
    b_std = []
    idx_max = []
    idx_min = []

    if method == "positive":
        check_first = 0
        check_last = 0

        while check_last < len(magnitudes) - 1:
            if check_last < check_first + n_b:
                check_last = check_first + n_b

            check_last += 1

            loop_mags = magnitudes[check_first : check_last + 1]  # noqa
            idx = np.argsort(times[check_first : check_last + 1])  # noqa
            loop_mags = loop_mags[idx]
            diffs = np.diff(loop_mags)

            while sum(diffs > 0) > n_b:
                loop_mags = magnitudes[check_first:check_last]
                idx = np.argsort(times[check_first : check_last + 1])  # noqa
                diffs = np.diff(loop_mags)
                check_first += 1

            if sum(diffs > 0) == n_b:
                b_loop, std_loop = estimate_b_positive(
                    loop_mags, delta_m=delta_m, return_std=True
                )
                b_any.append(b_loop)
                b_std.append(std_loop)
                idx_min.append(check_first)
                idx_max.append(check_last)
                check_first = int(
                    np.round(overlap * (check_first - check_last) + check_last)
                )

    elif method == "tinti":
        if mc is None:
            mc = magnitudes.min()
            print("no mc given, chose minimum magnitude of the sample")
        for ii in np.arange(
            0,
            n_eval,
            n_b - int(max(0, round(n_b * overlap - 1, 1))),
        ):
            loop_mags = magnitudes[ii : ii + n_b + 1]  # noqa
            idx = np.argsort(times[ii : ii + n_b + 1])  # noqa
            loop_mags = loop_mags[idx]

            b_loop, std_loop = estimate_b_tinti(
                loop_mags, mc=mc, delta_m=delta_m, return_std=True
            )
            b_any.append(b_loop)
            b_std.append(std_loop)

            idx_min.append(ii)
            idx_max.append(ii + n_b)

    b_any = np.array(b_any)
    b_std = np.array(b_std)
    idx_min = np.array(idx_min)
    idx_max = np.array(idx_max)

    if return_bar is True:
        out = (b_any, [idx_min, idx_max])
    else:
        out = (b_any, idx_max)

    if return_std is True:
        out = out + (b_std,)

    return out


# ========== need to comment still ========


def normalcdf_incompleteness(
    mags: np.ndarray, mc: float, sigma: float
) -> np.ndarray:
    """Filtering function: normal cdf with a standard deviation of sigma. The
    output can be interpreted as the probability to detect an earthquake. At
    mc, the probability of detect an earthquake is per definition 50%.

    Args:
        mags:

    """
    p = np.array(len(mags))
    x = (mags - mc) / sigma
    p = norm.cdf(x)
    return p


def distort_completeness(
    mags: np.ndarray, mc: float, sigma: float
) -> np.ndarray:
    p = normalcdf_incompleteness(mags, mc, sigma)
    p_test = np.random.rand(len(p))
    return mags[p > p_test]


# ========== need to comment still ========
