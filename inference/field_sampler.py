"""Batched field-level posterior for projected fields (Section 4 of the paper).

The posterior sampled here is

    -ln P(Delta_0 | Delta_obs) = 1/2 sum_k |Delta_0(k)|^2 / P_0(k)
                               + 1/2 sum_k |W.Delta_obs - W.(S * Z(Delta_0))|^2(k) / N(k) + const,
    N(k) = (1 - S^2(k)) P_tot(k),      S(k) = exp(-k^2 Sigma^2 / 2),

with P_0 (linear projected power), Sigma^2 and P_tot (total non-linear projected power) all
*fixed inputs*.  Nothing but the initial field is sampled, so there is no measured-power term
and the density is a proper posterior for the generative model Delta_obs = S Z(Delta_0) + n.

Differences with respect to `ProjectedFieldLikelihood` + `hmc_sample_z1.py`:
  * P_tot is an input (fiducial: mean power of the simulation suite), not a sampled parameter;
  * only modes inside the Nyquist circle (those present in the simulation ICs and constrained
    by prior and likelihood) are free; the corner modes and the DC mode are set to zero;
  * the field is parametrised by a real white field xi with Delta_0(k) = sqrt(P_0(k)) FFT(xi)/N,
    i.e. the sampler is preconditioned with the prior and has no flat directions;
  * P_0 and P_tot are evaluated per mode rather than per coarse k-bin;
  * the damping S is applied before the mask (the mask multiplies the predicted *mean* field);
  * the masked residual is weighted with N(k), not fsky N(k) (see `mask_noise`);
  * everything is batched, so B realizations (or B chains) are sampled in parallel.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference.projected_likelihood_torch import (  # noqa: E402
    FlatLambdaCDM,
    _lpt_displacements,
    build_k_grids,
    growth_factor,
    make_log_k_edges,
    projected_linear_power_2d,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("PROFIELD_DATA", Path(__file__).resolve().parents[1] / "data"))
RESULTS = DATA / "results"           # sampler outputs (~1 GB in total)
NG, LBOX, Z_INI = 128, 250.0, 63.0
KF = 2.0 * np.pi / LBOX
KNYQ = np.pi * NG / LBOX
NSIM = int(os.environ.get("PROFIELD_NSIM", 100))   # smaller only for quick tests


# --------------------------------------------------------------------------------------- data
def load_map(name: str, axis: str) -> tuple[np.ndarray, float]:
    d = np.load(DATA / "projected" / f"{name}.npz")
    return d[axis].astype(np.float64), float(d["redshift"])


def load_evolved(sims, axis: str) -> tuple[np.ndarray, float]:
    """z~1 projected fields of the given simulations along `axis`: (len(sims), N, N)."""
    out, zs = [], []
    for i in sims:
        f, z = load_map(f"sim{int(i):02d}_z1", axis)
        out.append(f)
        zs.append(z)
    # sims 00-49 and 50-99 use different output lists: never trust the snapshot index
    if max(zs) - min(zs) > 1e-3 or abs(zs[0] - 1.0) > 0.01:
        raise ValueError(f"evolved fields are not all at z~1: {min(zs)} .. {max(zs)}")
    return np.stack(out), zs[0]


def load_true_initial(sims, axis: str) -> np.ndarray:
    """True initial projected field at z=63.  Along z all simulations share the projected
    modes, and the truth is the projected-modes-only IC.  Along x or y every simulation has its
    own projected modes, taken from its z=63 output."""
    if axis == "z":
        f, _ = load_map("onlyPF_ics", "z")
        return np.repeat(f[None], len(sims), axis=0)
    out = []
    for i in sims:
        # sims 00-49 start their output list at z=49; their z=63 state is the separate IC file
        name = f"sim{int(i):02d}_ics" if (DATA / "projected" / f"sim{int(i):02d}_ics.npz").exists() else f"sim{int(i):02d}_z63"
        f, z = load_map(name, axis)
        if abs(z - Z_INI) > 0.01:
            raise ValueError(f"{name}: initial field is at z={z}, expected {Z_INI}")
        out.append(f)
    return np.stack(out)


def load_window(kind: str = "holes") -> np.ndarray:
    if kind == "none":
        return np.ones((NG, NG))
    return np.load(ROOT / "inference" / "window_holes.npz")["window"].astype(np.float64)


def kgrid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kf = 2.0 * np.pi * np.fft.fftfreq(NG, d=LBOX / NG)
    kx, ky = np.meshgrid(kf, kf, indexing="ij")
    return kx, ky, np.sqrt(kx**2 + ky**2)


def cic_window() -> np.ndarray:
    """Fourier window of cloud-in-cell assignment, sinc^2 per dimension."""
    kx, ky, _ = kgrid()
    c = LBOX / NG
    return (np.sinc(kx * c / (2 * np.pi)) * np.sinc(ky * c / (2 * np.pi))) ** 2


def fft_norm(field: np.ndarray) -> np.ndarray:
    return np.fft.fft2(field) * (LBOX / NG**2)


def shell_index() -> np.ndarray:
    """Index of the |k| shell (width k_f) of every mode."""
    return np.rint(kgrid()[2] / KF).astype(int)


def shell_average(power: np.ndarray, use: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Average a per-mode power (N,N) in shells of width k_f, optionally only over the modes
    flagged in `use`.  Returns (k_mean, P_shell)."""
    idx = shell_index().ravel()
    k = kgrid()[2].ravel()
    w = np.ones(idx.shape) if use is None else use.ravel().astype(float)
    n = np.maximum(np.bincount(idx, weights=w), 1e-30)
    return np.bincount(idx, weights=k * w) / n, np.bincount(idx, weights=power.ravel() * w) / n


def per_mode_from_shells(k_mean: np.ndarray, p_shell: np.ndarray) -> np.ndarray:
    """Interpolate a shell-averaged spectrum back onto the mode grid (log-log)."""
    kmag = kgrid()[2]
    out = np.zeros_like(kmag)
    good = (k_mean > 0) & (p_shell > 0)
    pos = kmag > 0
    out[pos] = np.exp(np.interp(np.log(kmag[pos]), np.log(k_mean[good]), np.log(p_shell[good])))
    return out


_SUITE_CACHE: dict = {}


def suite_mode_power() -> np.ndarray:
    """|FFT|^2 (physical units) of the x and y projections of all simulations at z~1:
    array (NSIM, 2, N, N).  These are 2 NSIM projected fields with independent projected modes."""
    if "p" not in _SUITE_CACHE:
        p = np.zeros((NSIM, 2, NG, NG))
        for i in range(NSIM):
            for j, ax in enumerate("xy"):
                p[i, j] = np.abs(fft_norm(load_map(f"sim{i:02d}_z1", ax)[0])) ** 2
        _SUITE_CACHE["p"] = p
    return _SUITE_CACHE["p"]


def suite_ptot(exclude: int | None = None) -> np.ndarray:
    """Fiducial P_tot(k) per mode: mean non-linear projected power of the simulation suite
    (x and y projections, optionally leaving out simulation `exclude`), averaged in |k| shells."""
    p = suite_mode_power()
    tot = p.sum(axis=(0, 1))
    n = 2 * NSIM
    if exclude is not None:
        tot = tot - p[int(exclude)].sum(axis=0)
        n -= 2
    # The x and y projections contain the line of modes with k_z = 0 (second index of the
    # stored maps), which belongs to the projected modes shared by all simulations and is
    # therefore not averaged over realizations.  Leave it out.
    independent = kgrid()[1] != 0
    return per_mode_from_shells(*shell_average(tot / n, use=independent))


def data_ptot(data: np.ndarray, window: np.ndarray, k_edges: np.ndarray) -> np.ndarray:
    """Plug-in alternative: binned power of the (masked) data itself, divided by fsky."""
    fsky = float((window**2).mean())
    kmag = kgrid()[2]
    p = np.abs(fft_norm(data * window)) ** 2 / fsky
    b = np.digitize(kmag.ravel(), k_edges) - 1
    ok = (b >= 0) & (b < len(k_edges) - 1) & (kmag.ravel() > 0)
    nb = len(k_edges) - 1
    pb = np.bincount(b[ok], weights=p.ravel()[ok], minlength=nb) / np.maximum(np.bincount(b[ok], minlength=nb), 1)
    return pb[np.clip(b, 0, nb - 1)].reshape(NG, NG)


def linear_p0_per_mode(z: float = Z_INI) -> np.ndarray:
    """Linear projected power P_0(k) per mode in physical units, (Mpc/h)^2."""
    kmag = kgrid()[2]
    out = np.zeros_like(kmag)
    pos = kmag > 0
    out[pos] = projected_linear_power_2d(kmag[pos], z, LBOX)
    return out


def bin_modes(k_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    kmag = kgrid()[2].ravel()
    b = np.digitize(kmag, k_edges) - 1
    ok = (b >= 0) & (b < len(k_edges) - 1) & (kmag > 0)
    return b, ok


# ------------------------------------------------------------------------------------ posterior
class FieldPosterior:
    """Batched posterior of the initial projected field given masked evolved fields."""

    def __init__(
        self,
        data: np.ndarray,            # (B, N, N) evolved projected fields
        window: np.ndarray,          # (N, N)
        z: float,
        sigma2: float,
        ptot: np.ndarray,            # (N, N) or (B, N, N) total power per mode, physical units
        *,
        lpt_order: int = 1,
        p0: np.ndarray | None = None,   # (N, N) linear power per mode, physical units
        damp_before_mask: bool = True,
        mask_noise: str = "plain",
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.device, self.dtype = torch.device(device), dtype
        self.cdtype = torch.complex64 if dtype == torch.float32 else torch.complex128
        t = lambda a: torch.as_tensor(np.asarray(a), device=self.device, dtype=dtype)  # noqa: E731
        self.B = data.shape[0]
        self.z, self.sigma2, self.lpt_order = float(z), float(sigma2), int(lpt_order)
        self.damp_before_mask = bool(damp_before_mask)
        self.growth = float(growth_factor(self.z, z_ini=Z_INI, cosmo=FlatLambdaCDM()))
        self.window = t(window)
        self.fsky = float((np.asarray(window) ** 2).mean())
        _, _, k2, kmag = build_k_grids(NG, LBOX, self.device, dtype)
        self.k2, self.kmag = k2, kmag
        self.valid = (kmag > 0) & (kmag <= KNYQ * (1 + 1e-6))
        self.S = torch.exp(-0.5 * k2 * self.sigma2)
        p0 = linear_p0_per_mode() if p0 is None else p0
        # Delta_0(k) [unnormalised fft2] = amp * fft2(xi), xi real white noise of unit variance.
        self.p0_raw = t(p0) * (NG**2 / LBOX) ** 2
        self.amp = torch.sqrt(self.p0_raw) / NG * self.valid.to(dtype)
        # Variance assigned to the modes of the masked residual W(x) n(x).
        #  "plain":     N(k).  chi^2 = sum_k |FFT(W r)|^2 / N(k) = (W r)^T C^-1 (W r), i.e. the
        #               residual is set to zero in the masked area.  Exact for white noise (it is
        #               then the sum over observed pixels); otherwise it ignores correlations of
        #               the residual across the mask boundary.
        #  "fsky":      fsky N(k), as in the previous version of the paper.  This is the variance
        #               of an individual masked mode, but dividing by it treats all N^2 masked
        #               modes as independent while only fsky N^2 pixels are observed, so the data
        #               are over-weighted by 1/fsky and the posterior is too narrow.
        #  "convolved": sum_k' |w(k-k')|^2 N(k'), the exact diagonal of the masked covariance
        #               (same over-weighting as "fsky").
        noise_full = (1.0 - self.S**2) * t(ptot)
        if mask_noise == "plain":
            noise = noise_full
        elif mask_noise == "fsky":
            noise = self.fsky * noise_full
        elif mask_noise == "convolved":
            w2 = (torch.fft.fft2(self.window) / NG**2).abs() ** 2
            noise = (NG**2 * torch.fft.fft2(torch.fft.ifft2(w2) * torch.fft.ifft2(noise_full))).real
        else:
            raise ValueError("mask_noise must be 'plain', 'fsky' or 'convolved'")
        self.mask_noise = mask_noise
        self.noise = noise.clamp_min(1e-12)
        self.data = t(data)
        self.data_k = self.fft_norm(self.data * self.window)

    # ---- forward model
    @staticmethod
    def fft_norm(field: torch.Tensor) -> torch.Tensor:
        return torch.fft.fft2(field) * (LBOX / NG**2)

    def modes(self, xi: torch.Tensor) -> torch.Tensor:
        return self.amp * torch.fft.fft2(xi)

    def initial_field(self, xi: torch.Tensor) -> torch.Tensor:
        return torch.fft.ifft2(self.modes(xi)).real

    def xi_from_field(self, field: torch.Tensor) -> torch.Tensor:
        """Whitened variable of a given initial field (invalid modes are dropped)."""
        inv = torch.where(self.amp > 0, 1.0 / self.amp.clamp_min(1e-30), torch.zeros_like(self.amp))
        return torch.fft.ifft2(torch.fft.fft2(field) * inv).real

    def evolve(self, field0: torch.Tensor) -> torch.Tensor:
        """Batched 2D LPT evolution (0 = Eulerian linear, 1 = Zeldovich, 2 = 2LPT)."""
        if self.lpt_order == 0:
            return self.growth * field0
        disp = _lpt_displacements(field0, lbox=LBOX, growth=self.growth, lpt_order=self.lpt_order)
        cell = LBOX / NG
        grid = torch.arange(NG, device=self.device, dtype=self.dtype)
        qx, qy = torch.meshgrid(grid, grid, indexing="ij")
        x = torch.remainder(qx * cell + disp[0], LBOX) / cell
        y = torch.remainder(qy * cell + disp[1], LBOX) / cell
        i0, j0 = torch.floor(x).long(), torch.floor(y).long()
        tx, ty = x - i0, y - j0
        i1, j1 = (i0 + 1) % NG, (j0 + 1) % NG
        i0, j0 = i0 % NG, j0 % NG
        nb = field0.shape[0]
        offs = (torch.arange(nb, device=self.device) * NG * NG).reshape(nb, 1, 1)
        flat = torch.zeros(nb * NG * NG, device=self.device, dtype=self.dtype)
        for w, ix, iy in (((1 - tx) * (1 - ty), i0, j0), ((1 - tx) * ty, i0, j1),
                          (tx * (1 - ty), i1, j0), (tx * ty, i1, j1)):
            flat.scatter_add_(0, (offs + ix * NG + iy).reshape(-1), w.reshape(-1))
        return flat.reshape(nb, NG, NG) - 1.0

    def mean_field(self, field0: torch.Tensor) -> torch.Tensor:
        """Unmasked model mean S * Z(Delta_0) in real space."""
        return torch.fft.ifft2(torch.fft.fft2(self.evolve(field0)) * self.S).real

    def predicted_k(self, field0: torch.Tensor) -> torch.Tensor:
        if self.damp_before_mask:
            return self.fft_norm(self.mean_field(field0) * self.window)
        return self.fft_norm(self.evolve(field0) * self.window) * self.S

    def chi2_modes(self, field0: torch.Tensor) -> torch.Tensor:
        """|residual|^2 / N per mode, zero outside the Nyquist circle: (B, N, N)."""
        r = self.data_k - self.predicted_k(field0)
        return (r.real**2 + r.imag**2) / self.noise * self.valid

    def log_like(self, field0: torch.Tensor) -> torch.Tensor:
        """Gaussian log-likelihood including the ln N normalisation: (B,).  With the "plain"
        mask weighting the chi^2 is a sum over the observed area, so only a fraction fsky of
        the modes is counted in the normalisation (this matters only when N is varied, e.g. in
        a scan over Sigma^2; the sampler works at fixed N)."""
        ndof = self.fsky if self.mask_noise == "plain" else 1.0
        lnn = ndof * (torch.log(self.noise) * self.valid).sum(dim=(-2, -1))
        return -0.5 * (self.chi2_modes(field0).sum(dim=(-2, -1)) + lnn)

    def potential(self, xi: torch.Tensor) -> torch.Tensor:
        """-ln posterior up to a constant, per batch element: (B,)."""
        return 0.5 * (xi**2).sum(dim=(-2, -1)) + 0.5 * self.chi2_modes(self.initial_field(xi)).sum(dim=(-2, -1))

    def potential_and_grad(self, xi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = xi.detach().requires_grad_(True)
        u = self.potential(x)
        (g,) = torch.autograd.grad(u.sum(), x)
        return u.detach(), g.detach()


# ------------------------------------------------------------------------------------------ HMC
def hmc(
    post: FieldPosterior,
    *,
    n_burn: int = 300,
    n_keep: int = 600,
    num_steps: int = 50,
    step0: float = 0.02,
    target_accept: float = 0.75,
    seed: int = 0,
    xi_init: torch.Tensor | None = None,
    callback=None,
    update=None,
    verbose: bool = True,
) -> dict:
    """Batched HMC with identity mass in the whitened variable and one step size per chain.
    Step sizes adapt during burn-in and are then frozen (with +-20% jitter per trajectory).
    `callback(it, xi)` is called for every kept iteration.  `update(it, xi)`, if given, is
    called after every iteration and may change the posterior (e.g. a Gibbs draw of other
    parameters); the potential is then re-evaluated."""
    gen = torch.Generator(device=post.device)
    gen.manual_seed(seed)
    B = post.B
    xi = torch.zeros(B, NG, NG, device=post.device, dtype=post.dtype) if xi_init is None else xi_init.clone()
    log_step = torch.full((B,), math.log(step0), device=post.device, dtype=post.dtype)
    log_step_hist = []
    u, g = post.potential_and_grad(xi)
    accept_hist, u_hist = [], []
    for it in range(n_burn + n_keep):
        jitter = 0.8 + 0.4 * torch.rand(B, device=post.device, dtype=post.dtype, generator=gen)
        eps = (torch.exp(log_step) * jitter).reshape(B, 1, 1)
        p = torch.randn(xi.shape, device=post.device, dtype=post.dtype, generator=gen)
        h0 = u + 0.5 * (p**2).sum(dim=(-2, -1))
        x_new, g_new = xi, g
        p = p - 0.5 * eps * g_new
        for s in range(num_steps):
            x_new = x_new + eps * p
            u_new, g_new = post.potential_and_grad(x_new)
            p = p - (0.5 if s == num_steps - 1 else 1.0) * eps * g_new
        dh = u_new + 0.5 * (p**2).sum(dim=(-2, -1)) - h0
        dh = torch.where(torch.isfinite(dh), dh, torch.full_like(dh, float("inf")))
        a_prob = torch.exp(-dh).clamp(max=1.0)
        acc = torch.rand(B, device=post.device, dtype=post.dtype, generator=gen) < a_prob
        m = acc.reshape(B, 1, 1)
        xi = torch.where(m, x_new, xi)
        g = torch.where(m, g_new, g)
        u = torch.where(acc, u_new, u)
        if update is not None:
            update(it, xi)
            u, g = post.potential_and_grad(xi)
        if it < n_burn:
            log_step = log_step + 0.1 * (a_prob - target_accept)
            log_step_hist.append(log_step.clone())
            if it == n_burn - 1:  # freeze at the average over the second half of burn-in
                log_step = torch.stack(log_step_hist[n_burn // 2:]).mean(dim=0)
        else:
            accept_hist.append(a_prob.cpu().numpy())
            u_hist.append(u.cpu().numpy())
            if callback is not None:
                callback(it - n_burn, xi)
        if verbose and it % max(1, (n_burn + n_keep) // 10) == 0:
            print(f"  iter {it:5d}  <accept>={float(a_prob.mean()):.2f}  <step>={float(torch.exp(log_step).mean()):.4f}"
                  f"  <U>={float(u.mean()):.1f}", flush=True)
    return {
        "xi": xi,
        "accept": np.array(accept_hist).mean(axis=0),
        "step": torch.exp(log_step).cpu().numpy(),
        "potential": np.array(u_hist),          # (n_keep, B)
    }


def ess(x: np.ndarray) -> float:
    """Effective sample size of a 1D chain (Geyer initial positive sequence)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    x = x - x.mean()
    var = x.var()
    if var == 0 or n < 8:
        return float(n)
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n] / (var * n)
    tau = 1.0
    for t in range(1, n - 1, 2):
        pair = acf[t] + acf[t + 1]
        if pair < 0:
            break
        tau += 2.0 * pair
    return float(n / max(tau, 1e-3))


def default_k_edges(nbins: int = 15) -> np.ndarray:
    return make_log_k_edges(NG, LBOX, nbins=nbins)
