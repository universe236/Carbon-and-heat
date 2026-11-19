"""Planting priorities compared at equal canopy area (Methods, 'Planting priorities compared', equation 4).

Plantable land in each 30 m pixel is its fraction of grass, shrubs, bare ground and non-building urban surfaces
(road space included, because canopy can extend over it); existing trees, water, buildings, crops and wetlands are
excluded. The canopy budget grows from one to ten canopy points, each one percentage point of the evaluation area,
capped by the plantable land.

Cooling-first planting ranks additions by their contribution to heat-weighted cooling across the whole temperature
field. An adjoint solve gives this contribution for every pixel at once:

    ψ − ∇·(L² ∇ψ) = D,        h_i = ψ_i (θ_eq,i − θ_eq,i⁺) / r_i                         (4)

ψ links a local temperature change to the surrounding excess heat D, θ_eq,i⁺ is the equilibrium temperature after
filling the remaining plantable fraction r_i of pixel i, and h_i is the cooling score per unit added canopy. Lengths
and scores are recomputed after each canopy point, so earlier additions change the priority of later planting.
Carbon-first planting ranks land by carbon accumulation per unit area; balanced planting gives the two initial
scores equal weight after dividing each by its city maximum; random placement averages three random orders. Every
plan is evaluated with the full temperature model at every budget.

The land surface is held in a state dict of 2-D arrays (fractions 0-1, NDVI, albedo). A user-supplied function
`features(state) -> (C, H, W) float32 array` turns a state into model predictors.
"""
import numpy as np
import torch

from .heat import excess_removed, heat_weighted_cooling
from .pde import solve

COVER = ('tree', 'shrub', 'grass', 'crop', 'wetland', 'built', 'bare', 'water', 'building', 'road')


def _get(state, key):
    return np.nan_to_num(np.asarray(state[key], dtype=np.float32))


def plantable_fraction(state, evaluation):
    """Plantable share of each pixel, zero outside the evaluation area."""
    building = np.minimum(_get(state, 'built'), _get(state, 'building'))
    pool = _get(state, 'grass') + _get(state, 'shrub') + _get(state, 'bare') + (_get(state, 'built') - building)
    free = 1 - _get(state, 'tree') - _get(state, 'water') - _get(state, 'building') - _get(state, 'crop') \
        - _get(state, 'wetland')
    return np.where(evaluation, np.clip(np.minimum(pool, free), 0, 1), 0).astype(np.float32)


def add_canopy(state, added, new_tree):
    """Convert the fraction `added` of each pixel to tree canopy. Grass, shrubs, bare ground, non-building urban
    surfaces and road space shrink in proportion; buildings, water, crops and wetlands are unchanged. NDVI and albedo
    move towards the new-tree values (`new_tree` = dict(ndvi=..., albedo=...))."""
    building = np.minimum(_get(state, 'built'), _get(state, 'building'))
    pool = _get(state, 'grass') + _get(state, 'shrub') + _get(state, 'bare') + (_get(state, 'built') - building)
    k = np.where(pool > 0, 1 - np.minimum(added, pool) / np.maximum(pool, 1e-9), 1.0)
    out = dict(state)
    for name in ('grass', 'shrub', 'bare', 'road'):
        out[name] = _get(state, name) * k
    out['built'] = building + (_get(state, 'built') - building) * k
    out['tree'] = _get(state, 'tree') + added
    for name in ('ndvi', 'albedo'):
        v = np.asarray(state[name], dtype=np.float32)
        out[name] = np.where(np.isfinite(v), v * (1 - added) + new_tree[name] * added, v)
    return out


def take_area(order, area, budget):
    """Fill an area budget in the given order; the last pixel is filled partially. Returns area taken per pixel."""
    cs = np.cumsum(area[order])
    k = int(np.searchsorted(cs, budget))
    taken = np.zeros_like(area)
    taken[order[:k]] = area[order[:k]]
    if k < len(order):
        taken[order[k]] = budget - (cs[k - 1] if k else 0.0)
    return taken


def concentration(gain_per_m2, area):
    """How strongly a benefit depends on location: area-weighted Gini coefficient of the gain per unit area, and the
    share of the total gain on the best 10% of plantable area."""
    v = np.maximum(np.where(np.isfinite(gain_per_m2), gain_per_m2, 0), 0)
    o = np.argsort(v)
    cw = np.cumsum(area[o]) / area.sum()
    cv = np.cumsum(v[o] * area[o]) / max((v * area).sum(), 1e-30)
    gini = float(1 - 2 * np.trapezoid(cv, cw))
    o = o[::-1]
    top = np.cumsum(area[o]) / area.sum() <= 0.1
    return {'gini': gini, 'top10_share': float((v[o] * area[o])[top].sum() / max((v * area).sum(), 1e-30))}


class CityPlanting:
    """Planting comparison for one city.

    model:       trained CoolingModel
    features:    callable, state -> (C, H, W) predictors
    state:       land-surface state dict
    evaluation:  bool (H, W), evaluation area
    excess:      baseline excess heat D (H, W), from heat.excess_heat
    length_scale, multiplier: from calibration
    new_tree:    dict(ndvi=..., albedo=...) of added canopy
    carbon_rate: annual carbon accumulation per hectare of added crown cover at each plantable pixel, (H, W) or
                 a flat array over plantable pixels (t C ha^-1 yr^-1)
    """

    def __init__(self, model, features, state, evaluation, excess, length_scale, multiplier, new_tree, carbon_rate,
                 pixel_area=900.0, device='cpu'):
        self.model, self.features, self.state = model.to(device).eval(), features, state
        self.excess, self.scale, self.multiplier = excess.astype(np.float32), float(length_scale), float(multiplier)
        self.new_tree, self.pixel_area, self.device = new_tree, float(pixel_area), device
        self.capacity = plantable_fraction(state, evaluation)
        self.cand = self.capacity > 1e-3
        self.rows, self.cols = np.nonzero(self.cand)
        self.area = self.capacity[self.cand].astype(np.float64) * self.pixel_area
        self.eval_area = float(evaluation.sum()) * self.pixel_area
        rate = np.asarray(carbon_rate, dtype=np.float64)
        self.carbon_per_m2 = (rate[self.cand] if rate.ndim == 2 else rate) / 1e4   # t C m^-2 yr^-1
        self.theta0 = self._anomaly(state)
        self._h1 = None

    # model fields -----------------------------------------------------------------------------------------------
    def _x(self, state):
        return torch.from_numpy(np.ascontiguousarray(self.features(state), dtype=np.float32))[None].to(self.device)

    @torch.no_grad()
    def _fields(self, state):
        x = self._x(state)
        return self.model.theta_eq(x), self.model.length(x, self.scale)

    @torch.no_grad()
    def _anomaly(self, state):
        te, length = self._fields(state)
        return solve(te.double(), length.double())[0].cpu().numpy()

    def _map(self, taken_area):
        m = np.zeros_like(self.capacity)
        m[self.rows, self.cols] = (taken_area / self.pixel_area).astype(np.float32)
        return m

    # cooling-first ------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def scores(self, taken_area):
        """h, equation (4): cooling score per m^2 of the remaining plantable area after `taken_area` is planted."""
        added = self._map(taken_area)
        rest = self.capacity - added
        now = add_canopy(self.state, added, self.new_tree)
        te_now, length = self._fields(now)
        te_full, _ = self._fields(add_canopy(self.state, added + rest, self.new_tree))
        d = torch.from_numpy(self.excess)[None].to(self.device)
        psi = solve(d.double(), length.double())
        gain = ((te_now - te_full).double() * psi)[0].cpu().numpy() * self.pixel_area
        r = rest[self.cand]
        return np.where(r > 1e-6, gain[self.cand] / np.maximum(r * self.pixel_area, 1e-9), -np.inf)

    def initial_scores(self):
        if self._h1 is None:
            h = self.scores(np.zeros_like(self.area))
            self._h1 = np.where(np.isfinite(h), h, 0.0)
        return self._h1

    def budgets(self, points=range(1, 11)):
        """Canopy budgets in m^2: one canopy point is one percentage point of the evaluation area."""
        return [min(p / 100 * self.eval_area, self.area.sum()) for p in points]

    def cooling_first(self, budgets):
        """Nested cooling-first plans, recomputing lengths and scores before each addition."""
        taken, plans = np.zeros_like(self.area), []
        for b in budgets:
            h = self.scores(taken)
            taken = taken + take_area(np.argsort(-h), self.area - taken, b - taken.sum())
            plans.append(taken.copy())
        return plans

    # fixed orders -------------------------------------------------------------------------------------------------
    def carbon_first(self, budgets):
        o = np.argsort(-self.carbon_per_m2)
        return [take_area(o, self.area, b) for b in budgets]

    def balanced(self, budgets):
        h, c = self.initial_scores(), self.carbon_per_m2
        score = 0.5 * h / max(h.max(), 1e-30) + 0.5 * c / max(c.max(), 1e-30)
        o = np.argsort(-score)
        return [take_area(o, self.area, b) for b in budgets]

    def random(self, budgets, n=3, seed=0):
        rng = np.random.default_rng(seed)
        orders = [rng.permutation(len(self.area)) for _ in range(n)]
        return [[take_area(o, self.area, b) for b in budgets] for o in orders]

    # evaluation ---------------------------------------------------------------------------------------------------
    def evaluate(self, taken_area):
        """Full-model outcome of a plan: heat-weighted cooling H, excess heat removed E, annual carbon C (t C)."""
        after = self._anomaly(add_canopy(self.state, self._map(taken_area), self.new_tree))
        u = self.multiplier * (self.theta0 - after)
        return {'H': heat_weighted_cooling(self.excess, u, self.pixel_area),
                'E': excess_removed(self.excess, u),
                'C': float((taken_area * self.carbon_per_m2).sum())}

    def compare(self, points=range(1, 11), n_random=3, seed=0):
        """All four priorities at every budget. Random outcomes are averaged over the random orders."""
        budgets = self.budgets(points)
        plans = {'cooling_first': self.cooling_first(budgets), 'carbon_first': self.carbon_first(budgets),
                 'balanced': self.balanced(budgets)}
        out = {k: [self.evaluate(t) for t in v] for k, v in plans.items()}
        rand = [[self.evaluate(t) for t in seq] for seq in self.random(budgets, n_random, seed)]
        out['random'] = [{k: float(np.mean([r[i][k] for r in rand])) for k in ('H', 'E', 'C')}
                         for i in range(len(budgets))]
        return {'budgets_m2': budgets, 'outcomes': out}

    def exchange_curve(self, budget, weights=np.linspace(0, 1, 101)):
        """Best achievable combinations of the two benefits (first-order): for each weight on cooling, rank land by
        the weighted, max-normalised initial scores and record (cooling score, carbon) of the plan."""
        h, c = self.initial_scores(), self.carbon_per_m2
        hn, cn = h / max(h.max(), 1e-30), c / max(c.max(), 1e-30)
        pts = []
        for w in weights:
            t = take_area(np.argsort(-(w * hn + (1 - w) * cn)), self.area, budget)
            pts.append((float((t * h).sum()), float((t * c).sum())))
        return np.array(pts)

    def concentration(self):
        return {'cooling': concentration(self.initial_scores(), self.area),
                'carbon': concentration(self.carbon_per_m2, self.area)}


def relative_outcomes(outcomes):
    """Each benefit relative to the priority that maximises it: cooling to cooling-first, carbon to carbon-first.
    Ratios need a positive reference and are NaN otherwise."""
    rel = {}
    for name, seq in outcomes.items():
        rel[name] = []
        for i, o in enumerate(seq):
            h_ref, c_ref = outcomes['cooling_first'][i]['H'], outcomes['carbon_first'][i]['C']
            rel[name].append({'cooling_kept': o['H'] / h_ref if h_ref > 0 else float('nan'),
                              'carbon_kept': o['C'] / c_ref if c_ref > 0 else float('nan')})
    return rel
