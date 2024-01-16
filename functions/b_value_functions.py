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


def b_val_dist_shibolt(x: np.ndarray, b: float, n: int):
    """Probability density of the estimated b-value for a given true b-value.
    The assumption used here is that the central limit theorem applies -
    this is valid only if the estimate is based on a large number of events
    (n>50 is used as a rule of thumb). Note: this works also for beta.

    Args:
        x:  b-value estimates
        b:  true b-value
        n:  number of events from which the b-value is estimated

    Returns:
        pdf:    probability density at x

    """
    pdf = (
        b / x**2 * np.sqrt(n / 2 / np.pi) * np.exp(-n / 2 * (1 - b / x) ** 2)
    )

    return pdf
