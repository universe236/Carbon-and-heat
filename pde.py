"""Diffusion–relaxation operator  A t = t − ∇·(L² ∇t)  on a regular grid, its solver and the sensor response.

The operator is discretised with finite volumes on a five-point stencil. Face coefficients are the harmonic mean of
L² on the two neighbouring cells divided by the cell size squared, and the outer boundary has zero flux. A is
symmetric positive definite, so the same solver gives the temperature field (eq. 2), the adjoint weight ψ used for
cooling-first planting (eq. 4) and the implicit gradient used in training.

All tensors are batched: (B, H, W). Lengths are in metres.
"""
import torch
import torch.nn.functional as F

CELL_M = 30.0


def face_coefficients(length, cell=CELL_M):
    k = length * length
    cx = 2 * k[..., :, 1:] * k[..., :, :-1] / (k[..., :, 1:] + k[..., :, :-1]) / cell ** 2
    cy = 2 * k[..., 1:, :] * k[..., :-1, :] / (k[..., 1:, :] + k[..., :-1, :]) / cell ** 2
    return cx, cy


def apply_operator(t, cx, cy):
    fx = cx * (t[..., :, 1:] - t[..., :, :-1])
    fy = cy * (t[..., 1:, :] - t[..., :-1, :])
    out = t.clone()
    out[..., :, :-1] -= fx
    out[..., :, 1:] += fx
    out[..., :-1, :] -= fy
    out[..., 1:, :] += fy
    return out


def _diagonal(cx, cy, shape):
    d = torch.ones(shape, dtype=cx.dtype, device=cx.device)
    d[..., :, :-1] += cx
    d[..., :, 1:] += cx
    d[..., :-1, :] += cy
    d[..., 1:, :] += cy
    return d


def conjugate_gradient(b, cx, cy, x0=None, tol=1e-6, maxiter=5000):
    """Jacobi-preconditioned conjugate gradient with a relative-residual stop per batch item."""
    dinv = 1.0 / _diagonal(cx, cy, b.shape)
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - apply_operator(x, cx, cy)
    z = dinv * r
    p = z.clone()
    rz = (r * z).sum(dim=(-2, -1), keepdim=True)
    bnorm = b.flatten(-2).norm(dim=-1).clamp_min(1e-30)
    for _ in range(maxiter):
        active = (r.flatten(-2).norm(dim=-1) / bnorm) >= tol
        if not active.any():
            break
        ap = apply_operator(p, cx, cy)
        alpha = rz / (p * ap).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-30)
        alpha = torch.where(active[..., None, None], alpha, torch.zeros_like(alpha))
        x = x + alpha * p
        r = r - alpha * ap
        z = dinv * r
        rz_new = (r * z).sum(dim=(-2, -1), keepdim=True)
        p = z + (rz_new / rz.clamp_min(1e-30)) * p
        rz = rz_new
    return x


class _Solve(torch.autograd.Function):
    """Linear solve with an implicit (adjoint) gradient, so training does not store solver iterations."""

    @staticmethod
    def forward(ctx, b, cx, cy, tol):
        with torch.no_grad():
            t = conjugate_gradient(b, cx, cy, tol=tol)
        ctx.save_for_backward(t, cx, cy)
        ctx.tol = tol
        return t

    @staticmethod
    def backward(ctx, grad):
        t, cx, cy = ctx.saved_tensors
        with torch.no_grad():
            lam = conjugate_gradient(grad.contiguous(), cx, cy, tol=ctx.tol)
            gcx = -(lam[..., :, 1:] - lam[..., :, :-1]) * (t[..., :, 1:] - t[..., :, :-1])
            gcy = -(lam[..., 1:, :] - lam[..., :-1, :]) * (t[..., 1:, :] - t[..., :-1, :])
        return lam, gcx, gcy, None


def solve(b, length, tol=1e-6):
    """Solve t − ∇·(L² ∇t) = b. Differentiable with respect to b and L."""
    cx, cy = face_coefficients(length)
    return _Solve.apply(b, cx, cy, tol)


def gaussian_kernel(fwhm_m=100.0, cell=CELL_M, device='cpu', dtype=torch.float32):
    """One-dimensional Gaussian kernel of the thermal sensor response (Landsat thermal band, 100 m)."""
    sigma = fwhm_m / 2.3548 / cell
    r = int(torch.ceil(torch.tensor(3 * sigma)).item())
    x = torch.arange(-r, r + 1, dtype=dtype, device=device)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def sensor_response(img, kernel):
    """Apply the separable Gaussian sensor response with reflect padding. img: (B, H, W)."""
    r = (kernel.numel() - 1) // 2
    x = F.pad(img[:, None], (r, r, 0, 0), mode='reflect')
    x = F.conv2d(x, kernel.view(1, 1, 1, -1))
    x = F.pad(x, (0, 0, r, r), mode='reflect')
    x = F.conv2d(x, kernel.view(1, 1, -1, 1))
    return x[:, 0]
