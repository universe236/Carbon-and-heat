# Planting for carbon pulls urban trees away from the heat — model code

This repository contains the models used in the paper: the cooling model and its calibration, the tree-growth and
carbon accumulation model, the comparison of planting priorities within cities and the allocation of a pooled canopy
budget among cities. Data preparation (downloading and gridding satellite, land-cover, building, terrain, climate and
emissions data) is not included; the code starts from prepared 30 m layers.

## Contents

| Module | Methods section | What it does |
|---|---|---|
| `treeplace/heat.py` | Surface temperature, high-temperature zones and excess heat | High-temperature zone (city mean + 1 s.d.), excess heat *D*, heat-weighted cooling *H* and fraction of excess heat removed *E* (eq. 1) |
| `treeplace/pde.py` | Cooling model | Operator θ − ∇·(L²∇θ), conjugate-gradient solver with implicit gradient, 100 m Gaussian sensor response |
| `treeplace/cooling.py` | Cooling model and calibration | T = T_b + θ, θ − ∇·(L²∇θ) = θ_eq with neural θ_eq and L (eq. 2); smooth city background; fit to observed surface temperature |
| `treeplace/calibration.py` | Cooling model and calibration | Cooling-length scaling to park profiles; event cooling as a 60–300 m ring contrast; placebo patches; common calibration multiplier; held-out within-city ranking test |
| `treeplace/carbon.py` | Tree carbon accumulation | Light-use-efficiency production (eq. 3) with CASA modifiers, new-tree greenness with β, IPCC-scaled carbon accumulation (gains only), share of local fossil emissions |
| `treeplace/beta.py` | Tree carbon accumulation | City fixed-effects estimate of β with city-clustered standard errors |
| `treeplace/planting.py` | Planting priorities compared | Plantable land, canopy budgets, adjoint cooling-first planting (eq. 4), carbon-first, balanced and random planting, full-model evaluation, exchange curve, concentration of gains |
| `treeplace/pooled.py` | Allocation among cities by carbon | Pooled budget filled by carbon accumulation rate across cities (eq. 5), recipients, cooling per unit canopy relative to each city's own cooling-first plan |

## Installation

```bash
pip install -r requirements.txt
```

Python 3.10 or later. A GPU is optional; pass `device='cuda'` where a function accepts it.

## Inputs

All inputs are prepared outside this package.

- **Land-surface state** of a city: a dict of 2-D arrays on a 30 m grid with cover fractions `tree`, `shrub`,
  `grass`, `crop`, `wetland`, `built`, `bare`, `water`, `building`, `road` (0–1), and `ndvi`, `albedo`.
- **Predictor function** `features(state) -> (C, H, W)` that turns a state into the model predictors (land cover,
  vegetation, buildings, terrain and climate). Planting changes the state; the function rebuilds the predictors.
- **Warm-season surface-temperature anomaly** and a validity mask, for fitting the cooling model and for excess heat.
- **Monthly climate** (12 values each): shortwave radiation, air temperature, precipitation, potential
  evapotranspiration and high-vegetation leaf area; and the **building-shade fraction** of each candidate pixel and
  month.
- **Observed tree-gain events**: observed and modelled event cooling, pre-gain surface temperature and city, for the
  calibration multiplier and the held-out ranking test; patch greenness and neighbourhood established-tree greenness
  for β.

## Workflow

```python
from treeplace import calibration, carbon, cooling, heat, planting, pooled

# 1. Cooling model (eq. 2): fit to observed warm-season surface temperature across cities
model = cooling.CoolingModel(theta_channels, length_channels, mean, std)
background = cooling.CityBackground(city_shapes)
cooling.train(model, background, cities)

# 2. Calibration: cooling length from park profiles, then one multiplier from observed tree gains
scale = calibration.length_scale(model_length, evaluation, park_length_m)
multiplier = calibration.calibration_multiplier(observed_event_cooling, modelled_event_cooling)
test = calibration.heldout_ranking(city, observed, modelled, pre_gain_lst)

# 3. Carbon accumulation rate of added canopy (eq. 3)
f_new = carbon.new_tree_fpar(f_city_event, f_local_existing, f_city_existing, beta=0.21)
npp_new = carbon.npp(shortwave, shade, f_new, t_month, t_opt, f_w, season)
rate = carbon.accumulation_rate(npp_new, carbon.ipcc_multiplier(npp_existing_crowns))

# 4. Planting priorities within a city (eq. 4)
D, _ = heat.excess_heat(lst_anomaly, valid)
city = planting.CityPlanting(model, features, state, evaluation, D, scale, multiplier, new_tree, rate)
result = city.compare(points=range(1, 11))
relative = planting.relative_outcomes(result['outcomes'])

# 5. Pooled allocation among cities by carbon (eq. 5)
alloc = pooled.pooled_allocation(city_list, budget_fraction=0.08)
```

`examples/quickstart.py` runs every step on a small random city to check the installation; its numbers have no
meaning.

```bash
python examples/quickstart.py
```

## Settings used in the paper

| Setting | Value |
|---|---|
| Grid | 30 m |
| High-temperature zone | city mean anomaly + 1 standard deviation |
| Sensor response | Gaussian, 100 m full width at half maximum |
| Cooling length | bounded 30–600 m, scaled to city park profiles (median 114 m) |
| Event contrast | unchanged pixels 60–300 m from the patch |
| Calibration multiplier | 3.33 |
| Light-use efficiency ε | 0.55 g C MJ⁻¹ |
| Neighbourhood greenness β | 0.21 (95% interval 0.17–0.25); neighbourhood of about 1 km |
| Carbon accumulation | IPCC rate 2.8 t C (ha crown cover)⁻¹ yr⁻¹ for existing crowns; multiplier 0.657; gains only |
| Moisture | rain-fed f_W = 0.5 + 0.5 min(1, P/PET); irrigated f_W = 1 |
| Canopy budgets | 1–10 percentage points of the evaluation area; 8 points is the reference |
| Random placement | mean of three random orders |
| Exchange curve | 101 weights |

## Citation

Please cite the paper when using this code (reference to be added on publication).

## License

To be added by the authors.
