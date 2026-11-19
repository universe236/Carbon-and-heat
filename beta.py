"""How strongly the greenness of new trees tracks that of nearby established trees (Methods, 'Tree carbon
accumulation').

New trees take time to mature, so β is estimated from observed tree-gain patches: the greenness (absorbed-radiation
fraction) of each patch is regressed on the mean greenness of established trees within about 1 km. The regression
uses city fixed effects, because carbon-first planting compares locations within a city and the estimate should come
from local contrasts rather than climatic differences between cities. The interval uses standard errors clustered by
city, because patches in the same city share climate and satellite observation conditions.
"""
import numpy as np


def fixed_effects_slope(y, x, city):
    """Within-city slope of y on x with a city-clustered standard error.
    Returns dict(beta, se, ci95, n, n_cities)."""
    y, x, city = np.asarray(y, float), np.asarray(x, float), np.asarray(city)
    yd, xd = y.copy(), x.copy()
    for c in np.unique(city):
        m = city == c
        yd[m] -= y[m].mean()
        xd[m] -= x[m].mean()
    sxx = float((xd ** 2).sum())
    beta = float((xd * yd).sum() / sxx)
    resid = yd - beta * xd
    score = np.array([(xd[city == c] * resid[city == c]).sum() for c in np.unique(city)])
    se = float(np.sqrt((score ** 2).sum()) / sxx)
    return {'beta': beta, 'se': se, 'ci95': (beta - 1.96 * se, beta + 1.96 * se), 'n': int(len(y)),
            'n_cities': int(len(np.unique(city)))}
