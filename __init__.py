"""Models for comparing where urban tree canopy is planted: cooling, tree carbon accumulation and allocation.

Modules follow the Methods of the accompanying paper:
    heat         high-temperature zones, excess heat, heat-weighted cooling (eq. 1)
    pde          diffusion–relaxation operator and solver used by eqs 2 and 4
    cooling      cooling model, sensor response and training loss (eq. 2)
    calibration  cooling-length scaling, temporal calibration, held-out ranking test
    carbon       light-use-efficiency growth model and IPCC-scaled carbon accumulation (eq. 3)
    beta         city fixed-effects estimate of the neighbourhood greenness coefficient
    planting     cooling-first, carbon-first, balanced and random planting (eq. 4)
    pooled       allocation of a pooled canopy budget among cities by carbon (eq. 5)
"""
__version__ = '1.0.0'
