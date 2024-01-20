# imports
import numpy as np
from seismostats import simulate_magnitudes, bin_to_precision


def transform_n(x: float, b: float, n1: int, n2: int):
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


def acf_lag_n(series: np.ndarray, lag: int = 1):
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
        # print(sum((b_series - np.mean(b_series)) ** 2), "sum")
        # print(len(b_series), np.mean(b_series), "len, mean")
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


def simulate_step(
    n: int,
    n_deviation: int,
    b: float,
    delta_b: float,
    mc: float,
    delta_m: float = 0.01,
):
    """Simulate binned magnitudes with a step of length N_deviation in the
    b-value

    Args:
        n:              total number of magnitudes to simulate
        n_deviation:    number of magnitudes with deviating b-value
        b:              b-value of the background
        delta_b:        deviation of b-value
        mc:             completeness magnitude
        delta_m:        magnitude bin width

    Returns:
        magnitudes:   array of magnitudes
        b_true: array of b-values from which each magnitude was simulated

    """
    n_loop1 = int((n - n_deviation) / 2)

    b_true = np.ones(n) * b
    b_true[n_loop1 : n_loop1 + n_deviation] = b + delta_b  # noqa

    magnitudes = simulated_magnitudes_binned(n, b, mc, delta_m)

    return magnitudes, b_true


def utsu_test(b1: float, b2: float, n1: int, n2: int):
    """Given two b-value estimates from two magnitude samples, this functions
    gives back the probability that the actual underlying b-values are not
    different. A small p-value means that the b-values are

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
