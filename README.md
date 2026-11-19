# Carbon and heat

Model code for the paper *Planting for carbon pulls urban trees away from the heat*.

The code compares where urban tree canopy is planted: for cooling, for carbon accumulation or at random. It covers
the cooling model and its calibration, the tree-growth model, the planting priorities within cities and the
allocation of a pooled canopy budget among cities. Data preparation is not included; the models start from prepared
30 m layers of each city.

## Contents

- `treeplace/cooling.py`, `pde.py`: cooling model
- `treeplace/calibration.py`: calibration against observed tree gains
- `treeplace/carbon.py`, `beta.py`: tree growth and carbon accumulation
- `treeplace/heat.py`, `planting.py`: planting priorities within a city
- `treeplace/pooled.py`: allocation among cities by carbon
- `examples/quickstart.py`: runs all steps on a small random city

## Getting started

```bash
pip install -r requirements.txt
python examples/quickstart.py
```

## Citation

Please cite the paper when using this code.
