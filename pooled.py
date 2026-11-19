"""Allocation of a pooled canopy budget among cities by carbon (Methods, 'Allocation among cities by carbon',
equation 5; Supplementary Note 3).

    B_pool = sum_j min(b A_j, P_j)
    max_x sum_i c_i x_i   subject to  sum_i x_i = B_pool,  0 <= x_i <= p_i

Each city contributes its nominal budget, capped by its plantable land, and the pool is planted on the land with the
highest carbon accumulation rate per unit area across all cities, working down the ranking until the area is used.
This is equivalent to one carbon threshold for all cities; a city receives no canopy when none of its land reaches
it. For recipients, cooling per unit canopy is compared with the city's own cooling-first plan under its original
budget, using the initial cooling scores on both sides (a first-order comparison).
"""
import numpy as np

from .planting import take_area


def pooled_allocation(cities, budget_fraction):
    """cities: list of dicts with
        name                 city identifier
        evaluation_area_m2   A_j
        area                 plantable area per pixel, m^2 (n,)
        carbon_per_m2        carbon accumulation rate per unit area (n,)
        score                initial cooling score per m^2, planting.CityPlanting.initial_scores() (n,)
    Returns dict(threshold, cities=[...]) with allocated area, recipient flag and relative cooling per m^2."""
    budgets = [min(budget_fraction * c['evaluation_area_m2'], float(np.sum(c['area']))) for c in cities]
    area = np.concatenate([np.asarray(c['area'], float) for c in cities])
    rate = np.concatenate([np.asarray(c['carbon_per_m2'], float) for c in cities])
    owner = np.concatenate([np.full(len(c['area']), j) for j, c in enumerate(cities)])
    order = np.argsort(-rate)
    taken = take_area(order, area, sum(budgets))
    filled = order[taken[order] > 0]
    threshold = float(rate[filled].min()) if len(filled) else float('nan')
    out = []
    for j, c in enumerate(cities):
        m = owner == j
        a_alloc = float(taken[m].sum())
        score = np.asarray(c['score'], float)
        h_alloc = float((taken[m] * score).sum())
        own = take_area(np.argsort(-score), np.asarray(c['area'], float), budgets[j])
        h_own = float((own * score).sum())
        rel = (h_alloc / a_alloc) / (h_own / budgets[j]) if a_alloc > 0 and h_own > 0 else float('nan')
        out.append({'name': c['name'], 'own_budget_m2': budgets[j], 'allocated_m2': a_alloc,
                    'receives': a_alloc > 0, 'cooling_per_m2_relative': rel})
    return {'threshold': threshold, 'pooled_budget_m2': float(sum(budgets)), 'cities': out}


def share_without_canopy(result, population=None):
    """Share of cities receiving no canopy and, if populations are given (same order), the share of people living
    in them."""
    none = np.array([not c['receives'] for c in result['cities']])
    out = {'cities': float(none.mean())}
    if population is not None:
        pop = np.asarray(population, float)
        out['population'] = float(pop[none].sum() / pop.sum())
    return out
