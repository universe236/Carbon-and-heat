"""Run every model component on toy inputs to check the installation.

The toy city below is random; the numbers printed have no meaning. Real use replaces it with prepared 30 m layers
of a city (land-cover fractions, NDVI, albedo, buildings, terrain, climate and warm-season surface temperature),
monthly climate, and the observed tree-gain events.
"""
import pathlib
import sys

import numpy as np
import torch
from scipy import ndimage

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # run from a clone without installing
from treeplace import beta, calibration, carbon, cooling, heat, planting, pooled  # noqa: E402

rng = np.random.default_rng(0)
torch.manual_seed(0)
H = W = 64


def smooth(a, s=3):
    return ndimage.gaussian_filter(a, s)


# ---------------------------------------------------------------- a toy city state (fractions sum to <= 1)
raw = {k: smooth(rng.random((H, W))) for k in ('tree', 'grass', 'built', 'bare', 'water')}
tot = sum(raw.values())
state = {k: (v / tot).astype(np.float32) for k, v in raw.items()}
state.update(shrub=np.zeros((H, W), np.float32), crop=np.zeros((H, W), np.float32),
             wetland=np.zeros((H, W), np.float32), road=0.2 * state['built'], building=0.5 * state['built'])
state['ndvi'] = (0.1 + 0.7 * state['tree'] + 0.4 * state['grass']).astype(np.float32)
state['albedo'] = (0.15 - 0.05 * state['tree'] + 0.05 * state['bare']).astype(np.float32)
evaluation = np.ones((H, W), bool)
NAMES = ['tree', 'grass', 'built', 'bare', 'water', 'building', 'road', 'ndvi', 'albedo']


def features(s):
    """Predictor stack of the toy model. Real predictors are prepared outside this package."""
    return np.stack([np.asarray(s[n], np.float32) for n in NAMES])


# ---------------------------------------------------------------- cooling model (eq. 2), short fit
x = torch.from_numpy(features(state))
observed = torch.from_numpy((3 * state['built'] - 4 * state['tree'] + 0.3 * rng.standard_normal((H, W))).astype(np.float32))
valid = torch.ones((H, W), dtype=torch.bool)
mean, std = x.mean(dim=(1, 2)), x.std(dim=(1, 2)) + 1e-6
model = cooling.CoolingModel(theta_channels=range(len(NAMES)), length_channels=[0, 2, 5], mean=mean, std=std)
background = cooling.CityBackground({'toy': (H, W)})
loss = cooling.train(model, background, {'toy': dict(x=x, lst=observed, valid=valid)}, steps=60)
print(f'cooling model: loss {loss[0]:.3f} -> {loss[-1]:.3f}')

# ---------------------------------------------------------------- calibration
with torch.no_grad():
    L = model.length(x[None])[0].numpy()
scale = calibration.length_scale(L, evaluation, park_length_m=114.0)
patch = np.zeros((H, W), bool)
patch[30:34, 30:34] = True
change = rng.normal(0, 0.3, (H, W))
change[patch] -= 1.5
print(f'event ring contrast {calibration.ring_contrast(change, patch, ~patch):.2f} °C; length scale {scale:.2f}')
n = 300
city = np.repeat(np.arange(10), n // 10)
pred = rng.random(n)
pre = rng.random(n)
obs = pred + 0.5 * pre + 0.3 * rng.standard_normal(n)
multiplier = calibration.calibration_multiplier(obs, pred)
test = calibration.heldout_ranking(city, obs, pred, pre)
print(f'multiplier {multiplier:.2f}; held-out within-city Spearman, median {test["median"]:.2f}')

# ---------------------------------------------------------------- carbon model (eq. 3) and beta
months = np.arange(12)
t_month = 15 + 10 * np.sin((months - 3) / 12 * 2 * np.pi)
lai = 1 + np.sin((months - 3) / 12 * 2 * np.pi)
shortwave = 400 + 200 * np.sin((months - 3) / 12 * 2 * np.pi)          # MJ m^-2 month^-1
precip, pet = np.full(12, 60.0), 40 + 40 * np.sin((months - 3) / 12 * 2 * np.pi) + 40
cap = planting.plantable_fraction(state, evaluation)
cand = cap > 1e-3
trees = state['tree'] >= np.quantile(state['tree'], 0.8)        # pixels standing in for established trees
city_ndvi, local_ndvi = carbon.established_tree_greenness(state['ndvi'], trees)
f_new = carbon.new_tree_fpar(carbon.fpar_from_ndvi(0.55), carbon.fpar_from_ndvi(local_ndvi[cand]),
                             float(carbon.fpar_from_ndvi(city_ndvi)))
args = dict(t_month=t_month, t_opt=carbon.optimum_temperature(t_month, lai),
            f_w=carbon.moisture_modifier(precip, pet), season=carbon.seasonality(lai))
npp_new = carbon.npp(shortwave, np.ones((12, cand.sum())), f_new, **args)
npp_old = carbon.npp(shortwave, np.ones((12, trees.sum())), carbon.fpar_from_ndvi(state['ndvi'][trees]), **args)
k = carbon.ipcc_multiplier(npp_old)
rate = np.zeros((H, W))
rate[cand] = carbon.accumulation_rate(npp_new, k)
gid = rng.integers(0, 20, 500)
xb = rng.random(500)
fit = beta.fixed_effects_slope(0.2 * xb + rng.normal(0, 0.1, 500), xb, gid)
print(f'IPCC multiplier {k:.3f}; beta {fit["beta"]:.2f} ({fit["ci95"][0]:.2f}-{fit["ci95"][1]:.2f})')

# ---------------------------------------------------------------- planting priorities (eq. 4)
lst = observed.numpy()
excess, thr = heat.excess_heat(lst, evaluation)
city_plan = planting.CityPlanting(model, features, state, evaluation, excess, scale, multiplier,
                                  new_tree={'ndvi': 0.55, 'albedo': 0.12}, carbon_rate=rate)
res = city_plan.compare(points=[1, 2, 4])
rel = planting.relative_outcomes(res['outcomes'])
for name in ('cooling_first', 'carbon_first', 'balanced', 'random'):
    r = rel[name][-1]
    print(f'{name:14s} cooling kept {r["cooling_kept"]:.2f}  carbon kept {r["carbon_kept"]:.2f}')
print('exchange curve points', city_plan.exchange_curve(res['budgets_m2'][-1]).shape,
      'concentration', {k: round(v['top10_share'], 2) for k, v in city_plan.concentration().items()})

# ---------------------------------------------------------------- pooled allocation (eq. 5)
base = dict(evaluation_area_m2=city_plan.eval_area, area=city_plan.area, score=city_plan.initial_scores())
cities = [dict(base, name='toy A', carbon_per_m2=city_plan.carbon_per_m2),
          dict(base, name='toy B', carbon_per_m2=0.5 * city_plan.carbon_per_m2)]
alloc = pooled.pooled_allocation(cities, budget_fraction=0.04)
print('pooled allocation:', [(c['name'], round(c['allocated_m2'] / 1e4, 1), c['receives']) for c in alloc['cities']],
      pooled.share_without_canopy(alloc))
