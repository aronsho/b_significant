# imports
import numpy as np


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
