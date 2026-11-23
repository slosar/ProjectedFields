from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FlatLambdaCDM:
    omega_m: float = 0.308
    omega_l: float = 0.692
    h: float = 0.678


def _as_real_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in (torch.float32, torch.complex64):
        return torch.float32
    if dtype in (torch.float64, torch.complex128):
        return torch.float64
    raise TypeError(f"Unsupported dtype: {dtype}")


def _as_complex_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in (torch.float32, torch.complex64):
        return torch.complex64
    if dtype in (torch.float64, torch.complex128):
        return torch.complex128
    raise TypeError(f"Unsupported dtype: {dtype}")


@lru_cache(maxsize=None)
def growth_factor(
    z: float,
    z_ini: float = 63.0,
    cosmo: FlatLambdaCDM | None = None,
    nsteps: int = 4096,
) -> float:
    """Return D(z) / D(z_ini) for a flat LCDM cosmology.

    Memoized: it is a pure function of its arguments but is called on every forward
    pass (via torch_evolve_lpt2d), where the 4096-point integration was wasted CPU work.
    FlatLambdaCDM is a frozen dataclass, so it is hashable and safe as a cache key."""
    cosmo = cosmo or FlatLambdaCDM()

    def hubble(a: np.ndarray) -> np.ndarray:
        return np.sqrt(cosmo.omega_m / a**3 + cosmo.omega_l)

    def linear_growth(a: float) -> float:
        a_grid = np.linspace(1e-4, a, nsteps)
        integrand = 1.0 / (a_grid**3 * hubble(a_grid) ** 3)
        return hubble(np.array([a]))[0] * np.trapezoid(integrand, a_grid)

    a = 1.0 / (1.0 + z)
    a_ini = 1.0 / (1.0 + z_ini)
    return linear_growth(a) / linear_growth(a_ini)


def sigma2_prior_mean_from_paper(
    z: float,
    z_ini: float = 63.0,
    sigma2_z0: float = 25.1,
    cosmo: FlatLambdaCDM | None = None,
) -> float:
    """Scale the L=250 Mpc/h value from the paper to the requested redshift."""
    g_z = growth_factor(z, z_ini=0.0, cosmo=cosmo)
    return sigma2_z0 * g_z**2


def efstathiou_transfer(k: np.ndarray, gamma: float = 0.21) -> np.ndarray:
    """BBKS / Efstathiou-Bond-White (1992) transfer function, k in h/Mpc.

    This is the shape-parameter spectrum GADGET-4 uses with PowerSpectrumType=0 +
    ShapeGamma (see data/param.txt: ShapeGamma=0.21), which generated the ICs.
    """
    q = k / gamma
    a, b, c, nu = 6.4, 3.0, 1.7, 1.13
    return (1.0 + (a * q + (b * q) ** 1.5 + (c * q) ** 2) ** nu) ** (-1.0 / nu)


def _linear_power_unnormalized(k: np.ndarray, ns: float = 1.0, gamma: float = 0.21) -> np.ndarray:
    return k ** ns * efstathiou_transfer(k, gamma=gamma) ** 2


def _sigma8_unnormalized(ns: float = 1.0, gamma: float = 0.21) -> float:
    k = np.logspace(-5, 2, 20000)
    pk = _linear_power_unnormalized(k, ns=ns, gamma=gamma)
    x = k * 8.0
    w = 3.0 * (np.sin(x) - x * np.cos(x)) / x ** 3
    integrand = k ** 2 * pk * w ** 2 / (2.0 * np.pi ** 2)
    return float(np.sqrt(np.trapezoid(integrand, k)))


def linear_power_3d(
    k: np.ndarray,
    z: float,
    *,
    sigma8: float = 0.9,
    ns: float = 1.0,
    gamma: float = 0.21,
    cosmo: FlatLambdaCDM | None = None,
) -> np.ndarray:
    """Efstathiou P_3D(k, z) in (Mpc/h)^3, normalized to sigma8 at z=0."""
    norm = (sigma8 / _sigma8_unnormalized(ns=ns, gamma=gamma)) ** 2
    pk0 = norm * _linear_power_unnormalized(np.asarray(k, dtype=float), ns=ns, gamma=gamma)
    growth = growth_factor(z, z_ini=0.0, cosmo=cosmo)  # D(z)/D(0)
    return pk0 * growth ** 2


def projected_linear_power_2d(
    k: np.ndarray,
    z: float,
    lbox: float,
    **kwargs: Any,
) -> np.ndarray:
    """2D power (Mpc/h)^2 of the projected linear field whose 3D power has support
    only in the k_z=0 plane (the fixed projected modes): P_2D(k_perp)=P_3D(k_perp)/L."""
    return linear_power_3d(k, z, **kwargs) / lbox


def ic_prior_power_bins(
    k_edges: np.ndarray,
    z: float,
    lbox: float,
    ngrid: int,
    **kwargs: Any,
) -> np.ndarray:
    """Analytic linear IC power evaluated at log-bin centers, converted to the
    convention used by the IC prior (unnormalized |fft2|^2 of the grid field).

    The prior compares |initial_modes()|^2 (unnormalized torch.fft.fft2) to P_ic,
    whereas physical 2D power uses fft2*(lbox/ngrid^2); the ratio is (ngrid^2/lbox)^2.
    """
    k_edges = np.asarray(k_edges, dtype=float)
    k_centers = np.sqrt(k_edges[:-1] * k_edges[1:])
    pk_phys = projected_linear_power_2d(k_centers, z, lbox, **kwargs)
    return pk_phys * (ngrid ** 2 / lbox) ** 2


def _fft2_norm(field: torch.Tensor, lbox: float) -> torch.Tensor:
    ngrid = field.shape[-1]
    return torch.fft.fft2(field) * (lbox / (ngrid**2))


def _ifft2_real(field_k: torch.Tensor) -> torch.Tensor:
    return torch.fft.ifft2(field_k).real


def _fft2_complex(field: torch.Tensor) -> torch.Tensor:
    return torch.fft.fft2(field.to(_as_complex_dtype(field.dtype)))


def build_k_grids(
    ngrid: int,
    lbox: float,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    real_dtype = _as_real_dtype(dtype)
    kf = 2.0 * np.pi * torch.fft.fftfreq(ngrid, d=lbox / ngrid, device=device)
    kx, ky = torch.meshgrid(kf, kf, indexing="ij")
    k2 = kx.square() + ky.square()
    kmag = torch.sqrt(k2)
    return kx.to(real_dtype), ky.to(real_dtype), k2.to(real_dtype), kmag.to(real_dtype)


def make_log_k_edges(
    ngrid: int,
    lbox: float,
    nbins: int = 20,
    kmin_factor: float = 1.0,
) -> np.ndarray:
    fundamental = 2.0 * np.pi / lbox
    nyquist = np.pi * ngrid / lbox
    return np.logspace(
        np.log10(fundamental * kmin_factor),
        np.log10(nyquist * (1.0 + 1e-6)),
        nbins + 1,
    )


def _bin_k_modes(
    kmag: torch.Tensor,
    k_edges: np.ndarray | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not torch.is_tensor(k_edges):
        k_edges = torch.as_tensor(k_edges, device=kmag.device, dtype=kmag.dtype)
    bin_index = torch.bucketize(kmag.reshape(-1), k_edges) - 1
    valid = (bin_index >= 0) & (bin_index < len(k_edges) - 1) & (kmag.reshape(-1) > 0)
    return bin_index, valid


def _binned_mode_mean(
    values: torch.Tensor,
    bin_index: torch.Tensor,
    valid_modes: torch.Tensor,
    nbins: int,
    floor: float,
) -> torch.Tensor:
    binned = torch.zeros(nbins, device=values.device, dtype=values.dtype)
    counts = torch.zeros(nbins, device=values.device, dtype=values.dtype)
    active_bins = bin_index[valid_modes]
    active_values = values.reshape(-1)[valid_modes]
    binned.scatter_add_(0, active_bins, active_values)
    counts.scatter_add_(0, active_bins, torch.ones_like(active_values))
    return binned / counts.clamp_min(1.0) + floor


def _cic_scatter_2d(
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    weights: torch.Tensor,
    ngrid: int,
) -> torch.Tensor:
    i0 = torch.floor(x_grid).to(torch.long)
    j0 = torch.floor(y_grid).to(torch.long)
    tx = x_grid - i0.to(x_grid.dtype)
    ty = y_grid - j0.to(y_grid.dtype)

    i1 = (i0 + 1) % ngrid
    j1 = (j0 + 1) % ngrid
    i0 = i0 % ngrid
    j0 = j0 % ngrid

    flat = torch.zeros(ngrid * ngrid, device=x_grid.device, dtype=x_grid.dtype)

    contributions = (
        ((1.0 - tx) * (1.0 - ty) * weights, i0, j0),
        ((1.0 - tx) * ty * weights, i0, j1),
        (tx * (1.0 - ty) * weights, i1, j0),
        (tx * ty * weights, i1, j1),
    )
    for value, ix, iy in contributions:
        flat.scatter_add_(0, ix * ngrid + iy, value)

    return flat.reshape(ngrid, ngrid)


def project_particles_cic_2d(
    positions: torch.Tensor,
    ngrid: int,
    lbox: float,
    axis: str = "z",
) -> torch.Tensor:
    """Project a 3D particle snapshot onto a 2D density-contrast mesh."""
    axis_map = {"x": (1, 2), "y": (0, 2), "z": (0, 1)}
    if axis not in axis_map:
        raise ValueError("axis must be one of 'x', 'y', or 'z'")

    dims = axis_map[axis]
    cell = lbox / ngrid
    xy = torch.remainder(positions[:, dims], lbox) / cell
    counts = _cic_scatter_2d(
        xy[:, 0],
        xy[:, 1],
        torch.ones(xy.shape[0], device=xy.device, dtype=xy.dtype),
        ngrid,
    )
    mean_count = positions.shape[0] / float(ngrid**2)
    return counts / mean_count - 1.0


def load_projected_field_from_snapshot(
    snapshot_path: str | Path,
    ngrid: int = 128,
    lbox: float = 250.0,
    axis: str = "z",
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, float]:
    snapshot_path = Path(snapshot_path)
    with h5py.File(snapshot_path, "r") as handle:
        positions = torch.as_tensor(
            handle["PartType1/Coordinates"][:],
            device=device,
            dtype=_as_real_dtype(dtype),
        )
        redshift = float(handle["Header"].attrs["Redshift"])
    field = project_particles_cic_2d(positions, ngrid=ngrid, lbox=lbox, axis=axis)
    return field.to(_as_real_dtype(dtype)), redshift


def build_tophat_window(
    ngrid: int,
    fill_fraction: float = 1.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Simple centered binary mask for end-to-end demos."""
    real_dtype = _as_real_dtype(dtype)
    if fill_fraction <= 0 or fill_fraction > 1:
        raise ValueError("fill_fraction must lie in (0, 1].")
    if fill_fraction == 1.0:
        return torch.ones((ngrid, ngrid), device=device, dtype=real_dtype)

    window = torch.zeros((ngrid, ngrid), device=device, dtype=real_dtype)
    width = max(1, int(round(fill_fraction * ngrid)))
    start = (ngrid - width) // 2
    stop = start + width
    window[start:stop, start:stop] = 1.0
    return window


def make_white_noise_initial_field(
    shape: tuple[int, int],
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
    std: float = 1.0,
) -> torch.Tensor:
    generator = torch.Generator(device=device if str(device) != "cpu" else "cpu")
    generator.manual_seed(seed)
    return std * torch.randn(shape, device=device, dtype=_as_real_dtype(dtype), generator=generator)


def _lpt_displacements(
    field_init: torch.Tensor,
    lbox: float,
    growth: float,
    lpt_order: int = 2,
) -> torch.Tensor:
    ngrid = field_init.shape[-1]
    complex_dtype = _as_complex_dtype(field_init.dtype)
    kx, ky, k2, _ = build_k_grids(ngrid, lbox, field_init.device, field_init.dtype)
    field_k = _fft2_complex(field_init)

    inv_k2 = torch.where(k2 > 0, 1.0 / k2, torch.zeros_like(k2))
    disp_x_k = 1j * kx * inv_k2 * field_k
    disp_y_k = 1j * ky * inv_k2 * field_k
    disp_x = growth * _ifft2_real(disp_x_k)
    disp_y = growth * _ifft2_real(disp_y_k)
    disp = torch.stack([disp_x, disp_y], dim=0)

    if lpt_order <= 1:
        return disp

    phi_xx = _ifft2_real(-(kx * kx * inv_k2).to(complex_dtype) * field_k)
    phi_yy = _ifft2_real(-(ky * ky * inv_k2).to(complex_dtype) * field_k)
    phi_xy = _ifft2_real(-(kx * ky * inv_k2).to(complex_dtype) * field_k)
    phi2 = phi_xx * phi_yy - phi_xy.square()
    phi2_k = torch.fft.fft2(phi2.to(complex_dtype))
    disp2_x = (3.0 / 7.0) * growth**2 * _ifft2_real(1j * kx * inv_k2 * phi2_k)
    disp2_y = (3.0 / 7.0) * growth**2 * _ifft2_real(1j * ky * inv_k2 * phi2_k)
    return disp + torch.stack([disp2_x, disp2_y], dim=0)


def torch_evolve_lpt2d(
    field_init: torch.Tensor,
    lbox: float,
    z: float,
    z_ini: float = 63.0,
    cosmo: FlatLambdaCDM | None = None,
    lpt_order: int = 2,
) -> torch.Tensor:
    """Differentiable 2D LPT evolution using a CIC particle sheet.

    lpt_order: 0 = Eulerian linear (just D(z)*delta_init, no displacement),
    1 = Zeldovich, 2 = 2LPT."""
    if field_init.ndim != 2 or field_init.shape[0] != field_init.shape[1]:
        raise ValueError("field_init must be a square 2D tensor")

    ngrid = field_init.shape[0]
    cell = lbox / ngrid
    growth = growth_factor(z, z_ini=z_ini, cosmo=cosmo)
    if lpt_order == 0:
        # Eulerian linear: delta(x,t) = D(t) delta_init(x); no displacement at all.
        return growth * field_init
    disp = _lpt_displacements(field_init, lbox=lbox, growth=growth, lpt_order=lpt_order)

    grid = torch.arange(ngrid, device=field_init.device, dtype=field_init.dtype)
    qx, qy = torch.meshgrid(grid, grid, indexing="ij")
    x = torch.remainder(qx * cell + disp[0], lbox) / cell
    y = torch.remainder(qy * cell + disp[1], lbox) / cell

    mesh = _cic_scatter_2d(
        x.reshape(-1),
        y.reshape(-1),
        torch.ones(ngrid * ngrid, device=field_init.device, dtype=field_init.dtype),
        ngrid,
    )
    return mesh - 1.0


class ProjectedFieldLikelihood(torch.nn.Module):
    def __init__(
        self,
        data_field: torch.Tensor,
        window: torch.Tensor,
        lbox: float,
        z: float,
        k_edges: np.ndarray | torch.Tensor,
        *,
        initial_field_init: torch.Tensor | None = None,
        initial_modes_init: torch.Tensor | None = None,
        sigma2_init: float | None = None,
        pk_init: torch.Tensor | np.ndarray | None = None,
        pic_init: torch.Tensor | np.ndarray | None = None,
        fit_ic_prior: bool = False,
        fit_pic: bool = True,
        tie_residual_to_total: bool = False,
        sigma2_prior_mean: float | None = None,
        sigma2_prior_std: float | None = None,
        log_pk_smoothness: float = 0.0,
        log_pic_smoothness: float = 0.0,
        pk_floor: float = 1e-6,
        lpt_order: int = 2,
        z_ini: float = 63.0,
        cosmo: FlatLambdaCDM | None = None,
    ) -> None:
        super().__init__()
        if data_field.shape != window.shape:
            raise ValueError("data_field and window must have identical shapes")

        self.lbox = float(lbox)
        self.z = float(z)
        self.z_ini = float(z_ini)
        self.lpt_order = int(lpt_order)
        self.cosmo = cosmo or FlatLambdaCDM()
        self.pk_floor = float(pk_floor)
        self.fit_ic_prior = bool(fit_ic_prior)
        self.fit_pic = bool(fit_pic)
        self.tie_residual_to_total = bool(tie_residual_to_total)
        self.log_pk_smoothness = float(log_pk_smoothness)
        self.log_pic_smoothness = float(log_pic_smoothness)
        self.sigma2_prior_mean = sigma2_prior_mean
        self.sigma2_prior_std = sigma2_prior_std

        real_dtype = _as_real_dtype(data_field.dtype)
        self.register_buffer("data_field", data_field.to(real_dtype))
        self.register_buffer("window", window.to(real_dtype))
        self.ngrid = data_field.shape[0]
        # Sky/area fraction for the pseudo-spectrum mask correction P_true ~ P_coupled/fsky
        # (mean(W^2) = unmasked fraction for a 0/1 mask; = 1 when unmasked).
        self.fsky = float((self.window ** 2).mean())

        _, _, k2, kmag = build_k_grids(self.ngrid, self.lbox, data_field.device, real_dtype)
        self.register_buffer("k2", k2)
        self.register_buffer("kmag", kmag)

        k_edges_tensor = torch.as_tensor(k_edges, device=data_field.device, dtype=real_dtype)
        self.register_buffer("k_edges", k_edges_tensor)
        bin_index, valid_modes = _bin_k_modes(kmag, k_edges_tensor)
        self.register_buffer("bin_index", bin_index)
        self.register_buffer("valid_modes", valid_modes)

        data_k = _fft2_norm(self.data_field * self.window, self.lbox)
        self.register_buffer("data_k", data_k)

        if initial_modes_init is None:
            if initial_field_init is None:
                initial_field_init = torch.zeros_like(self.data_field)
            initial_modes_init = _fft2_complex(initial_field_init.to(real_dtype))
        else:
            initial_modes_init = torch.as_tensor(
                initial_modes_init,
                device=data_field.device,
                dtype=_as_complex_dtype(real_dtype),
            )
        if pk_init is None:
            pk_init = torch.ones(len(k_edges_tensor) - 1, device=data_field.device, dtype=real_dtype)
        else:
            pk_init = torch.as_tensor(pk_init, device=data_field.device, dtype=real_dtype)
        if self.fit_ic_prior:
            if pic_init is None:
                pic_init = _binned_mode_mean(
                    initial_modes_init.abs().square().to(real_dtype),
                    bin_index,
                    valid_modes,
                    len(k_edges_tensor) - 1,
                    self.pk_floor,
                )
            else:
                pic_init = torch.as_tensor(pic_init, device=data_field.device, dtype=real_dtype)
        if sigma2_init is None:
            sigma2_init = sigma2_prior_mean if sigma2_prior_mean is not None else 10.0

        self.raw_initial_modes_real = torch.nn.Parameter(initial_modes_init.real.clone().to(real_dtype))
        self.raw_initial_modes_imag = torch.nn.Parameter(initial_modes_init.imag.clone().to(real_dtype))
        self.raw_sigma2 = torch.nn.Parameter(
            self._inverse_softplus(torch.as_tensor(sigma2_init, device=data_field.device, dtype=real_dtype))
        )
        self.raw_pk = torch.nn.Parameter(self._inverse_softplus(pk_init))
        if self.fit_ic_prior and self.fit_pic:
            self.raw_pic = torch.nn.Parameter(self._inverse_softplus(pic_init))
            self.register_buffer("pic_fixed", None)
        elif self.fit_ic_prior:
            # Frozen IC prior: P_ic held fixed (e.g. the analytic linear spectrum).
            self.raw_pic = None
            self.register_buffer("pic_fixed", pic_init.to(real_dtype))
        else:
            self.raw_pic = None
            self.register_buffer("pic_fixed", None)

        if self.tie_residual_to_total:
            nbins = len(k_edges_tensor) - 1
            data_power = self.data_k.abs().square()
            # Deconvolve the mask: the masked data power is ~fsky * true power.
            phat_total = _binned_mode_mean(data_power, bin_index, valid_modes, nbins, 0.0) / self.fsky
            mode_counts = torch.zeros(nbins, device=data_field.device, dtype=real_dtype)
            mode_counts.scatter_add_(
                0, bin_index[valid_modes], torch.ones(int(valid_modes.sum()), device=data_field.device, dtype=real_dtype)
            )
            self.register_buffer("phat_total", phat_total)
            self.register_buffer("mode_counts", mode_counts)
            self.raw_ptot = torch.nn.Parameter(self._inverse_softplus(phat_total.clamp_min(self.pk_floor)))
        else:
            self.register_buffer("phat_total", None)
            self.register_buffer("mode_counts", None)
            self.raw_ptot = None

    @staticmethod
    def _inverse_softplus(x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, min=1e-6)
        return torch.where(x > 20.0, x, torch.log(torch.expm1(x)))

    def sigma2(self) -> torch.Tensor:
        return F.softplus(self.raw_sigma2) + self.pk_floor

    def pk_bins(self) -> torch.Tensor:
        return F.softplus(self.raw_pk) + self.pk_floor

    def pic_bins(self) -> torch.Tensor | None:
        if self.raw_pic is not None:
            return F.softplus(self.raw_pic) + self.pk_floor
        if getattr(self, "pic_fixed", None) is not None:
            return self.pic_fixed
        return None

    def ptot_bins(self) -> torch.Tensor | None:
        if self.raw_ptot is None:
            return None
        return F.softplus(self.raw_ptot) + self.pk_floor

    def initial_modes(self) -> torch.Tensor:
        modes = torch.complex(self.raw_initial_modes_real, self.raw_initial_modes_imag)
        ngrid = modes.shape[0]
        device = modes.device
        neg_index = torch.remainder(-torch.arange(ngrid, device=device), ngrid).to(torch.long)
        partner = modes.index_select(0, neg_index).index_select(1, neg_index)
        return 0.5 * (modes + partner.conj())

    def initial_field(self) -> torch.Tensor:
        return _ifft2_real(self.initial_modes())

    def damped_initial_field(self) -> torch.Tensor:
        """Initial field low-pass filtered by the signal amplitude S=exp(-1/2 k^2 Sigma^2).
        Suppresses the poorly-constrained high-k modes that otherwise dominate the
        raw fitted field visually."""
        return _ifft2_real(self.initial_modes() * self.signal_fraction().sqrt())

    def evolved_field(self) -> torch.Tensor:
        return torch_evolve_lpt2d(
            self.initial_field(),
            lbox=self.lbox,
            z=self.z,
            z_ini=self.z_ini,
            cosmo=self.cosmo,
            lpt_order=self.lpt_order,
        )

    def predicted_k(self) -> torch.Tensor:
        evolved = self.evolved_field()
        pred_k = _fft2_norm(evolved * self.window, self.lbox)
        # The mean evolved field (paper Eq. key) damps the LPT *amplitude* by
        # S = exp(-1/2 k^2 Sigma^2) = sqrt(signal_fraction()).
        damping = self.signal_fraction().sqrt()
        return pred_k * damping

    def signal_fraction(self) -> torch.Tensor:
        # Surviving *power* fraction S^2 = exp(-k^2 Sigma^2); the field amplitude
        # is damped by S = sqrt of this (see predicted_k).
        return torch.exp(-self.k2 * self.sigma2())

    def residual_fraction(self) -> torch.Tensor:
        # Lost power fraction (1 - S^2).
        return torch.clamp(1.0 - self.signal_fraction(), min=self.pk_floor)

    def mode_power(self) -> torch.Tensor:
        pk = self.pk_bins()
        flat = pk[self.bin_index.clamp(min=0, max=len(pk) - 1)]
        return flat.reshape(self.ngrid, self.ngrid)

    def total_mode_power(self) -> torch.Tensor | None:
        ptot = self.ptot_bins()
        if ptot is None:
            return None
        flat = ptot[self.bin_index.clamp(min=0, max=len(ptot) - 1)]
        return flat.reshape(self.ngrid, self.ngrid)

    def noise_power(self) -> torch.Tensor:
        """Per-mode residual variance N(k). Either the free P(k) bins, or, when
        tying to the total, N(k) = (1 - S(k)^2) * P_total(k)."""
        if not self.tie_residual_to_total:
            return self.mode_power()
        # Residual carries the lost power fraction (1 - S^2) of the (true) total; the data
        # is masked, so the variance of the masked residual modes is fsky times that.
        return self.fsky * self.residual_fraction() * self.total_mode_power()

    def ptot_measurement_loglike(self) -> torch.Tensor:
        """Mode-counting (Gamma/Wishart) likelihood anchoring P_total(k) to the
        measured binned data power, with the correct sampling uncertainty
        sigma_P(k) = P_total(k)/sqrt(N_modes(k))."""
        ptot = self.ptot_bins()
        if ptot is None:
            return torch.zeros((), device=self.data_field.device, dtype=self.data_field.dtype)
        active = self.mode_counts > 0
        return -0.5 * torch.sum(
            self.mode_counts[active] * (self.phat_total[active] / ptot[active] + torch.log(ptot[active]))
        )

    def ic_mode_power(self) -> torch.Tensor | None:
        pic = self.pic_bins()
        if pic is None:
            return None
        flat = pic[self.bin_index.clamp(min=0, max=len(pic) - 1)]
        return flat.reshape(self.ngrid, self.ngrid)

    def log_likelihood(self) -> torch.Tensor:
        pred_k = self.predicted_k()
        resid = self.data_k - pred_k
        power = self.noise_power().reshape(-1)[self.valid_modes]
        resid2 = resid.reshape(-1)[self.valid_modes].abs().square()
        return -0.5 * torch.sum(resid2 / power + torch.log(power))

    def log_ic_prior(self) -> torch.Tensor:
        pic = self.ic_mode_power()
        if pic is None:
            return torch.zeros((), device=self.data_field.device, dtype=self.data_field.dtype)
        ic_modes2 = self.initial_modes().reshape(-1)[self.valid_modes].abs().square()
        ic_power = pic.reshape(-1)[self.valid_modes]
        return -0.5 * torch.sum(ic_modes2 / ic_power + torch.log(ic_power))

    def forward(self) -> torch.Tensor:
        """Enables torch.func.functional_call(model, params) -> loss, used by HMC."""
        return self.negative_log_posterior()

    def negative_log_posterior(self) -> torch.Tensor:
        loss = -(self.log_likelihood() + self.log_ic_prior() + self.ptot_measurement_loglike())

        if self.sigma2_prior_mean is not None and self.sigma2_prior_std is not None:
            sigma2 = self.sigma2()
            loss = loss + 0.5 * ((sigma2 - self.sigma2_prior_mean) / self.sigma2_prior_std) ** 2

        if not self.tie_residual_to_total and self.log_pk_smoothness > 0.0 and self.raw_pk.numel() > 1:
            log_pk = torch.log(self.pk_bins())
            loss = loss + self.log_pk_smoothness * torch.mean(torch.diff(log_pk).square())

        if self.raw_pic is not None and self.log_pic_smoothness > 0.0 and self.raw_pic.numel() > 1:
            log_pic = torch.log(self.pic_bins())
            loss = loss + self.log_pic_smoothness * torch.mean(torch.diff(log_pic).square())

        return loss

    @torch.no_grad()
    def diagnostics(self) -> dict[str, Any]:
        pred_k = self.predicted_k()
        resid = self.data_k - pred_k
        resid_power = resid.abs().square()
        pic = self.pic_bins()
        ptot = self.ptot_bins()
        return {
            "sigma2": float(self.sigma2().cpu()),
            "pk_bins": self.pk_bins().detach().cpu().numpy(),
            "pic_bins": None if pic is None else pic.detach().cpu().numpy(),
            "ptot_bins": None if ptot is None else ptot.detach().cpu().numpy(),
            "phat_total": None if not self.tie_residual_to_total else self.phat_total.detach().cpu().numpy(),
            "ptot_measurement_loglike": float(self.ptot_measurement_loglike().cpu()),
            "signal_fraction_mean": float(self.signal_fraction().mean().cpu()),
            "residual_fraction_mean": float(self.residual_fraction().mean().cpu()),
            "initial_modes_rms": float(torch.sqrt(self.initial_modes().abs().square().mean()).cpu()),
            "loss": float(self.negative_log_posterior().cpu()),
            "log_likelihood": float(self.log_likelihood().cpu()),
            "log_ic_prior": float(self.log_ic_prior().cpu()),
            "residual_rms_k": float(torch.sqrt(resid_power.mean()).cpu()),
        }


def fit_likelihood(
    model: ProjectedFieldLikelihood,
    *,
    steps: int = 500,
    lr: float = 5e-2,
    optimizer_name: str = "adam",
    report_every: int = 25,
) -> dict[str, Any]:
    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    else:
        raise ValueError("optimizer_name must be one of 'adam' or 'adamw'")

    history: dict[str, list[float]] = {"step": [], "loss": [], "sigma2": [], "log_likelihood": []}
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = model.negative_log_posterior()
        loss.backward()
        optimizer.step()

        if step % report_every == 0 or step == steps - 1:
            with torch.no_grad():
                ll = model.log_likelihood()
                history["step"].append(float(step))
                history["loss"].append(float(loss.detach().cpu()))
                history["sigma2"].append(float(model.sigma2().detach().cpu()))
                history["log_likelihood"].append(float(ll.detach().cpu()))

    return {
        "model": model,
        "history": history,
        "diagnostics": model.diagnostics(),
    }


def save_result_bundle(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_result_bundle(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location, weights_only=False)
