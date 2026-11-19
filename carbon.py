"""Tree growth and carbon accumulation of added canopy (Methods, equation 3; Supplementary Note 2).

    NPP_i = sum_m (0.5 R_m) s_im F_im ε f_T,m f_W,m

R_m is monthly shortwave radiation, s_im the fraction remaining after building shade, F_im the fraction of
photosynthetically active radiation absorbed by vegetation, ε the light-use efficiency, and f_T and f_W temperature
and moisture modifiers (CASA). Monthly leaf area supplies the seasonality of F.

New-tree greenness is anchored to the city's observed tree gains and adds a fraction β of the difference between
neighbourhood and citywide established-tree greenness:

    F_i,new = F_city,event + β (F_local,existing − F_city,existing),     β = 0.21

Production is converted to carbon accumulation with one multiplier that aligns the median for existing crowns with
the IPCC rate for urban trees (2.8 t C per ha of crown cover per year). Following the IPCC gain approach for trees in
settlements, only gains are counted: neither the vegetation replaced by new trees nor losses from mortality, pruning
and removal are deducted.

Monthly arrays have length 12; per-pixel arrays are flat over the candidate pixels of one city.
"""
import numpy as np
from scipy import ndimage

EPSILON = 0.55          # g C per MJ of absorbed photosynthetically active radiation
BETA = 0.21             # neighbourhood greenness coefficient (95% interval 0.17-0.25)
IPCC_RATE = 2.8         # t C (ha crown cover)^-1 yr^-1, IPCC 2019 Refinement, Vol. 4, Ch. 8, Table 8.1
SR_MIN, SR_MAX = 1.08, 6.17
FPAR_MIN, FPAR_MAX = 0.001, 0.95


def fpar_from_ndvi(ndvi, sr_min=SR_MIN, sr_max=SR_MAX):
    """Absorbed-radiation fraction from NDVI through the simple ratio SR = (1 + NDVI) / (1 − NDVI)."""
    v = np.asarray(ndvi, float)
    sr = (1 + v) / np.clip(1 - v, 1e-3, None)
    f = (sr - sr_min) / (sr_max - sr_min) * (FPAR_MAX - FPAR_MIN) + FPAR_MIN
    return np.clip(f, FPAR_MIN, FPAR_MAX)


def established_tree_greenness(ndvi, tree_mask, window_px=33):
    """Citywide median NDVI of established trees and its neighbourhood mean within a window of about 1 km.
    Returns (city value, neighbourhood map)."""
    city = float(np.median(ndvi[tree_mask]))
    num = ndimage.uniform_filter(np.where(tree_mask, ndvi, 0.0).astype(float), window_px)
    den = ndimage.uniform_filter(tree_mask.astype(float), window_px)
    local = np.where(den > 1e-6, num / np.maximum(den, 1e-9), city)
    return city, local


def new_tree_fpar(fpar_city_event, fpar_local_existing, fpar_city_existing, beta=BETA):
    """F_i,new. beta = 0 gives one value per city; beta = 1 passes the full neighbourhood difference to new trees."""
    return fpar_city_event + beta * (np.asarray(fpar_local_existing) - fpar_city_existing)


def optimum_temperature(t_month, leaf_area_month):
    """CASA optimum temperature: air temperature of the month with maximum high-vegetation leaf area."""
    return float(np.asarray(t_month)[int(np.argmax(leaf_area_month))])


def temperature_modifier(t, t_opt):
    """CASA temperature modifier f_T (t in °C)."""
    t = np.asarray(t, float)
    t1 = 0.8 + 0.02 * t_opt - 0.0005 * t_opt ** 2
    t2 = 1.1814 / (1 + np.exp(0.2 * (t_opt - 10 - t))) / (1 + np.exp(0.3 * (-t_opt - 10 + t)))
    return np.where(t > -10, t1 * t2, 0.0)


def moisture_modifier(precip, pet, irrigated=False):
    """f_W = 0.5 + 0.5 min(1, P/PET) when rain-fed; 1 when irrigation removes water stress."""
    if irrigated:
        return np.ones(len(precip))
    return 0.5 + 0.5 * np.minimum(1.0, np.asarray(precip, float) / np.maximum(np.asarray(pet, float), 1e-6))


def seasonality(leaf_area_month):
    lai = np.asarray(leaf_area_month, float)
    return lai / lai.max() if lai.max() > 0 else np.ones(12)


def npp(shortwave, shade, fpar, t_month, t_opt, f_w, season, epsilon=EPSILON, surface_anomaly=None):
    """Annual net primary production per unit crown area (g C m^-2 yr^-1), equation (3).

    shortwave: monthly shortwave radiation, MJ m^-2 (12,)
    shade:     fraction remaining after building shade, (12, n)
    fpar:      absorbed fraction of the new trees, (n,)
    t_month:   monthly air temperature, °C (12,)
    f_w:       monthly moisture modifier (12,)
    season:    monthly leaf-area seasonality (12,)
    surface_anomaly: optional (n,) warm-season surface-temperature anomaly. When given, the temperature modifier
                     uses f_T(T_m + ΔT_i), the within-city temperature case of Supplementary Note 5.
    """
    par = 0.5 * np.asarray(shortwave, float)
    t = np.asarray(t_month, float)[:, None]
    if surface_anomaly is not None:
        t = t + np.asarray(surface_anomaly, float)[None, :]
    ft = temperature_modifier(t, t_opt)
    if ft.ndim == 1:
        ft = ft[:, None]
    k = (par * np.asarray(season) * epsilon * np.asarray(f_w))[:, None]
    return (k * shade * ft * np.asarray(fpar)[None, :]).sum(0)


def ipcc_multiplier(npp_existing_crowns, rate=IPCC_RATE):
    """Multiplier that aligns the median production of existing tree crowns with the IPCC carbon accumulation rate.
    NPP in g C m^-2 yr^-1; 0.01 converts it to t C ha^-1 yr^-1."""
    return float(rate / (0.01 * np.median(npp_existing_crowns)))


def accumulation_rate(npp_new, multiplier):
    """CRW_i, annual carbon accumulation per hectare of added crown cover (t C ha^-1 yr^-1)."""
    return 0.01 * multiplier * np.asarray(npp_new, float)


def carbon_accumulation(added_area_ha, rate):
    """C = sum_i A_i CRW_i, annual tree carbon accumulation of added canopy (t C yr^-1). Gains only."""
    return float(np.sum(np.asarray(added_area_ha, float) * np.asarray(rate, float)))


def emissions_share(carbon_t, fossil_emissions_t):
    """Annual tree carbon accumulation as a share of local fossil carbon emissions (both t C yr^-1)."""
    return float(carbon_t / fossil_emissions_t)
