"""High-temperature zones, excess heat and the two cooling objectives (Methods, equation 1).

    H = sum_i a_i D_i u_i
    E = sum_i a_i min(max(u_i, 0), D_i) / sum_i a_i D_i

D_i is baseline excess heat (surface temperature above the city threshold), u_i is modelled cooling and a_i is
pixel area. H gives more weight to cooling hotter land and defines the cooling-first objective; E is the fraction of
excess heat removed and gives no credit for cooling below the threshold.
"""
import numpy as np


def high_temperature_threshold(lst_anomaly, valid):
    """Threshold of the high-temperature zone: city mean anomaly plus one standard deviation."""
    v = lst_anomaly[valid]
    return float(v.mean() + v.std())


def excess_heat(lst_anomaly, valid, threshold=None):
    """Baseline excess heat D: temperature above the threshold, zero elsewhere. Returns (D, threshold)."""
    if threshold is None:
        threshold = high_temperature_threshold(lst_anomaly, valid)
    d = np.where(valid & (lst_anomaly > threshold), lst_anomaly - threshold, 0.0)
    return d.astype(np.float32), threshold


def heat_weighted_cooling(excess, cooling, pixel_area=900.0):
    """H, equation (1)."""
    return float(np.sum(excess * cooling) * pixel_area)


def excess_removed(excess, cooling):
    """E, equation (1). NaN when the city has no excess heat."""
    hot = excess > 0
    if not hot.any():
        return float('nan')
    return float(np.minimum(np.maximum(cooling[hot], 0.0), excess[hot]).sum() / excess[hot].sum())
