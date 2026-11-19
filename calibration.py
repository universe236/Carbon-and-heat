"""Cooling-length scaling, temporal calibration against observed tree gains, and the held-out ranking test
(Methods, 'Cooling model and calibration').

Event cooling is a before–after contrast with reference locations: the temperature change in unchanged pixels
60–300 m from a tree-gain patch minus the change in the patch. The same contrast is applied to observed surface
temperatures, to model temperatures (after the sensor response) and to placebo patches of unchanged land. The ratio of
summed observed to summed modelled event cooling gives one multiplier for all cities; it changes absolute cooling but
not location rankings or ratios between planting priorities.
"""
import numpy as np
from scipy import ndimage, stats


def length_scale(model_length, mask, park_length_m):
    """Factor that scales modelled cooling length so its median over the evaluation area matches the city's park
    profile estimate (or its climate-group estimate where parks are insufficient)."""
    return float(park_length_m / np.median(model_length[mask]))


def ring_contrast(change, patch, unchanged, inner_m=60.0, outer_m=300.0, cell=30.0, min_pixels=10):
    """Cooling of one patch: mean change on unchanged pixels in the reference ring minus mean change on the patch.
    change = after − before. Returns NaN when the ring holds fewer than min_pixels unchanged pixels."""
    dist = ndimage.distance_transform_edt(~patch) * cell
    ring = unchanged & (dist > inner_m) & (dist <= outer_m)
    if ring.sum() < min_pixels:
        return float('nan')
    return float(np.nanmean(change[ring]) - np.nanmean(change[patch]))


def event_cooling(change, labels, unchanged, **ring):
    """Ring contrast for every labelled patch (labels > 0). Returns {label: cooling}."""
    out = {}
    for lab in np.unique(labels[labels > 0]):
        v = ring_contrast(change, labels == lab, unchanged, **ring)
        if np.isfinite(v):
            out[int(lab)] = v
    return out


def placebo_patches(unchanged, n, rng, size=3):
    """Random size × size blocks of unchanged land, used as patches with no tree gain."""
    core = ndimage.binary_erosion(unchanged, np.ones((size, size)))
    cand = np.argwhere(core)
    if not len(cand):
        return []
    pick = cand[rng.choice(len(cand), min(n, len(cand)), replace=False)]
    h = size // 2
    out = []
    for r, c in pick:
        m = np.zeros(unchanged.shape, bool)
        m[r - h:r + h + 1, c - h:c + h + 1] = True
        out.append(m)
    return out


def calibration_multiplier(observed, modelled):
    """Common multiplier: sum of observed over sum of modelled event cooling."""
    observed, modelled = np.asarray(observed, float), np.asarray(modelled, float)
    ok = np.isfinite(observed) & np.isfinite(modelled)
    return float(observed[ok].sum() / modelled[ok].sum())


def _centre(v, city):
    v = np.asarray(v, float)
    out = v.copy()
    for c in np.unique(city):
        m = city == c
        out[m] = v[m] - v[m].mean()
    return out


def _within_city_spearman(obs, pred, city):
    rho = {}
    for c in np.unique(city):
        m = city == c
        if m.sum() >= 3 and np.ptp(pred[m]) > 0 and np.ptp(obs[m]) > 0:
            rho[c] = stats.spearmanr(obs[m], pred[m]).statistic
    return rho


def _fit(obs, pred_c, heat_c):
    """Least-squares coefficients of city-centred observed cooling on city-centred model prediction and pre-gain
    surface temperature."""
    a = np.column_stack([pred_c, heat_c])
    coef, *_ = np.linalg.lstsq(a, obs, rcond=None)
    return coef


def heldout_ranking(city, observed, modelled, pre_gain_lst, weights=np.linspace(0, 1, 11)):
    """Within-city ranking test with each city held out in turn.

    The main model's calibration stays fixed. Using events from the other cities only, a linear correction to
    modelled cooling is fitted from city-centred model predictions and pre-gain surface temperature, and the weight
    of the correction is chosen by leave-one-city-out validation within those cities. Returns the Spearman
    correlation between predicted and observed cooling in each held-out city, its median, and the share of cities
    with a positive correlation.
    """
    city = np.asarray(city)
    obs, pred, heat = (np.asarray(v, float) for v in (observed, modelled, pre_gain_lst))
    obs_c, pred_c, heat_c = _centre(obs, city), _centre(pred, city), _centre(heat, city)
    cities = np.unique(city)

    def predict(train, test, w):
        a, b = _fit(obs_c[train], pred_c[train], heat_c[train])
        return a * pred_c[test] + w * b * heat_c[test]

    rho = {}
    for h in cities:
        test, train = city == h, city != h
        inner = cities[cities != h]
        score = []
        for w in weights:
            p = np.empty(train.sum())
            sub = city[train]
            for g in inner:
                p[sub == g] = predict(train & (city != g), train & (city == g), w)
            r = _within_city_spearman(obs[train], p, sub)
            score.append(np.median(list(r.values())) if r else -np.inf)
        w_best = weights[int(np.argmax(score))]
        r = _within_city_spearman(obs[test], predict(train, test, w_best), city[test])
        if h in r:
            rho[h] = r[h]
    vals = np.array(list(rho.values()))
    return {'per_city': rho, 'median': float(np.median(vals)) if len(vals) else float('nan'),
            'share_positive': float(np.mean(vals > 0)) if len(vals) else float('nan')}
