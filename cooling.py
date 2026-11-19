"""Cooling model (Methods, equation 2).

    T = T_b + θ,      θ − ∇·(L² ∇θ) = θ_eq

T is surface temperature, T_b a smooth city background, θ the local anomaly, θ_eq the local equilibrium temperature
and L the cooling length. The diffusion term carries a local temperature change into neighbouring land and the
relaxation term limits its reach. θ_eq and L are pixelwise neural functions of land cover, vegetation, buildings,
terrain and climate. Modelled temperatures pass through a 100 m Gaussian sensor response before they are compared
with Landsat observations.

Inputs x are (B, C, H, W) tensors of predictors on a 30 m grid. Which channels feed θ_eq and which feed L is set when
the model is built; building the predictor stack from raw data is outside this package.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pde import gaussian_kernel, sensor_response, solve


def _pixel_mlp(n_in, n_out, width=64):
    return nn.Sequential(nn.Conv2d(n_in, width, 1), nn.SiLU(), nn.Conv2d(width, width, 1), nn.SiLU(),
                         nn.Conv2d(width, n_out, 1))


class CoolingModel(nn.Module):
    def __init__(self, theta_channels, length_channels, mean, std, width=64, l_min=30.0, l_max=600.0,
                 sensor_fwhm_m=100.0):
        """theta_channels / length_channels: indices of the predictors used for θ_eq and L.
        mean, std: per-channel normalisation (length C). l_min, l_max: bounds of the cooling length (m)."""
        super().__init__()
        self.theta_channels = list(theta_channels)
        self.length_channels = list(length_channels)
        self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer('std', torch.as_tensor(std, dtype=torch.float32))
        self.theta_net = _pixel_mlp(len(self.theta_channels), 1, width)
        self.length_net = _pixel_mlp(len(self.length_channels), 1, width)
        self.l_min, self.l_max = float(l_min), float(l_max)
        self.register_buffer('kernel', gaussian_kernel(sensor_fwhm_m))

    def _norm(self, x, idx):
        return (x[:, idx] - self.mean[idx, None, None]) / self.std[idx, None, None]

    def theta_eq(self, x):
        return self.theta_net(self._norm(x, self.theta_channels))[:, 0]

    def length(self, x, scale=1.0):
        """Cooling length in metres; scale is the city factor from park temperature profiles."""
        g = self.length_net(self._norm(x, self.length_channels))[:, 0]
        return scale * (self.l_min + (self.l_max - self.l_min) * torch.sigmoid(g))

    def anomaly(self, x, scale=1.0, tol=1e-6):
        """θ from equation (2)."""
        return solve(self.theta_eq(x), self.length(x, scale), tol)

    def forward(self, x, background, scale=1.0, tol=1e-6):
        """Sensor-level surface temperature PSF ∗ (T_b + θ), with the intermediate fields."""
        te, length = self.theta_eq(x), self.length(x, scale)
        th = solve(te, length, tol)
        return {'lst': sensor_response(background + th, self.kernel), 'theta': th, 'theta_eq': te, 'L': length}


class CityBackground(nn.Module):
    """Smooth background T_b for each city: values on a coarse node grid (default 2 km), bilinearly interpolated."""

    def __init__(self, city_shapes, node_m=2000.0, cell=30.0):
        super().__init__()
        self.nodes = nn.ParameterDict({
            str(c): nn.Parameter(torch.zeros(max(2, round(h * cell / node_m) + 1), max(2, round(w * cell / node_m) + 1)))
            for c, (h, w) in city_shapes.items()})

    def forward(self, city, shape):
        return F.interpolate(self.nodes[str(city)][None, None], size=shape, mode='bilinear', align_corners=True)[0, 0]

    def roughness(self):
        """Mean squared second difference of the node grids; penalising it keeps T_b smooth."""
        out = 0.0
        for p in self.nodes.values():
            if p.shape[0] >= 3:
                out = out + ((p[2:] - 2 * p[1:-1] + p[:-2]) ** 2).mean()
            if p.shape[1] >= 3:
                out = out + ((p[:, 2:] - 2 * p[:, 1:-1] + p[:, :-2]) ** 2).mean()
        return out / len(self.nodes)


def fit_loss(model, background, city, x, observed, valid, smooth_weight=0.1, delta=1.0):
    """Huber misfit between sensor-level model output and observed warm-season surface temperature, over valid
    pixels, plus a smoothness penalty on the background. x: (1, C, H, W); observed, valid: (H, W)."""
    tb = background(city, observed.shape)[None]
    out = model(x, tb)
    w = valid.float()[None]
    misfit = (F.huber_loss(out['lst'], observed[None], reduction='none', delta=delta) * w).sum() / w.sum().clamp_min(1)
    return misfit + smooth_weight * background.roughness()


def train(model, background, cities, steps=3000, lr=3e-3, lr_background=3e-2, smooth_weight=0.1, seed=0,
          device='cpu'):
    """Fit the model to observed surface temperatures. cities: {city: dict(x=(C,H,W), lst=(H,W), valid=(H,W))}
    as tensors. One city is drawn per step."""
    torch.manual_seed(seed)
    model.to(device)
    background.to(device)
    opt = torch.optim.AdamW([{'params': model.parameters(), 'lr': lr},
                             {'params': background.parameters(), 'lr': lr_background, 'weight_decay': 0.0}],
                            weight_decay=1e-4)
    names = list(cities)
    gen = torch.Generator().manual_seed(seed)
    history = []
    for step in range(steps):
        c = names[int(torch.randint(len(names), (1,), generator=gen))]
        d = cities[c]
        loss = fit_loss(model, background, c, d['x'][None].to(device), d['lst'].to(device), d['valid'].to(device),
                        smooth_weight)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(background.parameters()), 5.0)
        opt.step()
        history.append(float(loss))
    return history


@torch.no_grad()
def cooling_map(model, x_before, x_after, scale, multiplier=1.0, tol=1e-6):
    """Calibrated cooling u = multiplier · (θ_before − θ_after) for one city. Inputs (C, H, W); returns (H, W)."""
    t0 = model.anomaly(x_before[None], scale, tol)
    t1 = model.anomaly(x_after[None], scale, tol)
    return (multiplier * (t0 - t1))[0]
