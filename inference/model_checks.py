"""Model checks at the true initial conditions (no sampling).

Uses the 100 z-projected realizations (same projected modes, different bulk modes) and the true
projected initial field to test the ingredients of the likelihood:

  1. the inputs: linear prior power P_0 against the true initial fields, and the suite-mean
     total power P_tot;
  2. the mean model: does S Z(Delta_0) reproduce the ensemble-mean evolved field, and is the
     residual uncorrelated with the prediction;
  3. the noise model: residual power against N = (1 - S^2) P_tot;
  4. the diagonal-covariance approximation: dependence of the residual variance on the local
     projected density and the off-diagonal Fourier correlations of the residual;
  5. the Sigma^2 scan of the likelihood at fixed (true) initial conditions.

Writes $PROFIELD_DATA/results/checks.npz and prints a summary.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference import field_sampler as fs  # noqa: E402

np.set_printoptions(precision=3, suppress=True, linewidth=220)
SIMS = list(range(fs.NSIM))
K_EDGES = fs.default_k_edges(15)
B_IDX, B_OK = fs.bin_modes(K_EDGES)
NB = len(K_EDGES) - 1
KMAG = fs.kgrid()[2]
VALID = (KMAG > 0) & (KMAG <= fs.KNYQ * (1 + 1e-6))
K_MEAN = np.bincount(B_IDX[B_OK], weights=KMAG.ravel()[B_OK], minlength=NB) / np.bincount(B_IDX[B_OK], minlength=NB)


def binned(per_mode: np.ndarray) -> np.ndarray:
    """Average a (..., N, N) per-mode quantity in the k-bins."""
    flat = per_mode.reshape(*per_mode.shape[:-2], -1)[..., B_OK]
    out = np.zeros(per_mode.shape[:-2] + (NB,))
    cnt = np.bincount(B_IDX[B_OK], minlength=NB)
    for b in range(NB):
        out[..., b] = flat[..., B_IDX[B_OK] == b].sum(axis=-1) / cnt[b]
    return out


def main() -> None:
    out: dict = {"k_mean": K_MEAN, "k_edges": K_EDGES}
    nshow = 11
    print("k bins          ", K_MEAN[:nshow])

    # ------------------------------------------------------------------ 1. inputs
    cic = fs.cic_window()
    p0 = fs.linear_p0_per_mode()
    t_z = fs.load_true_initial([0], "z")[0]
    tk_z = fs.fft_norm(t_z) / cic * VALID
    ic_pow = []
    for ax in "xy":
        tr = fs.load_true_initial(SIMS, ax)
        ic_pow.append(np.abs(fs.fft_norm(tr) / cic) ** 2)
    ic_pow = np.concatenate(ic_pow)                      # 200 projected ICs
    # modes with k_z = 0 (second index) are shared by all simulations: leave them out
    indep = (fs.kgrid()[1] != 0).astype(float)
    out["ic_power_ratio"] = binned(ic_pow.mean(axis=0) * indep) / binned(p0 * indep)
    out["ic_power_ratio_z"] = binned(np.abs(tk_z) ** 2) / binned(p0)
    print("P_IC(x,y sims)/P_0", out["ic_power_ratio"][:nshow])
    print("P_IC(z)/P_0       ", out["ic_power_ratio_z"][:nshow])
    # the z=63 output of a full simulation projected along z against the projected-modes-only IC
    name = "sim88_z63"
    s88 = fs.fft_norm(fs.load_map(name, "z")[0]) / cic * VALID
    out["ic_full_vs_pmo_r"] = binned((s88 * tk_z.conj()).real) / np.sqrt(binned(np.abs(s88) ** 2) * binned(np.abs(tk_z) ** 2))
    print("r(full sim z=63, PMO IC)", out["ic_full_vs_pmo_r"][:nshow])

    d_z, z = fs.load_evolved(SIMS, "z")
    dk = fs.fft_norm(d_z)                                # (100, N, N)
    ptot = fs.suite_ptot()
    pz = (np.abs(dk) ** 2).mean(axis=0)
    out.update(ptot_suite=binned(ptot), ptot_zsuite=binned(pz), z=z,
               k_shell=fs.shell_average(ptot)[0], ptot_shell=fs.shell_average(ptot)[1])
    print("P_tot(z suite)/P_tot(x,y suite)", (out["ptot_zsuite"] / out["ptot_suite"])[:nshow])

    # ------------------------------------------------------------------ 2-4. residual at truth
    truth = torch.as_tensor(np.fft.ifft2(np.fft.fft2(t_z) / cic * VALID).real[None], dtype=torch.float64)
    mean_dk = dk.mean(axis=0)
    for order, sigma2 in ((1, 8.2), (2, 8.2), (0, 8.2)):
        post = fs.FieldPosterior(d_z[:1], fs.load_window("none"), z, sigma2, ptot, lpt_order=order, dtype=torch.float64)
        mu = post.mean_field(truth)[0].numpy()
        muk = fs.fft_norm(mu)
        s2 = post.S.numpy() ** 2
        res = dk - muk                                    # (100, N, N)
        noise = (1 - s2) * ptot
        tag = f"lpt{order}"
        out[f"mean_slope_{tag}"] = binned((mean_dk * muk.conj()).real) / binned(np.abs(muk) ** 2)
        out[f"mean_r_{tag}"] = binned((mean_dk * muk.conj()).real) / np.sqrt(binned(np.abs(mean_dk) ** 2) * binned(np.abs(muk) ** 2))
        out[f"res_over_noise_{tag}"] = binned((np.abs(res) ** 2).mean(axis=0)) / binned(noise)
        out[f"res_mean_part_{tag}"] = binned(np.abs(res.mean(axis=0)) ** 2) / binned(noise)
        out[f"res_var_part_{tag}"] = binned(res.var(axis=0)) / binned(noise)
        # correlation coefficient of each realization's residual with the prediction
        rc = binned((res * muk.conj()).real) / np.sqrt(binned(np.abs(res) ** 2) * binned(np.abs(muk) ** 2))
        out[f"res_pred_r_{tag}"] = rc.mean(axis=0)
        out[f"res_pred_r_err_{tag}"] = rc.std(axis=0) / np.sqrt(len(SIMS))
        print(f"--- {tag}, Sigma2={sigma2}")
        print("  <d>.mu/|mu|^2   ", out[f"mean_slope_{tag}"][:nshow])
        print("  r(<d>, mu)      ", out[f"mean_r_{tag}"][:nshow])
        print("  <|res|^2>/N     ", out[f"res_over_noise_{tag}"][:nshow])
        print("    mean part     ", out[f"res_mean_part_{tag}"][:nshow])
        print("    variance part ", out[f"res_var_part_{tag}"][:nshow])
        print("  r(res, mu)      ", out[f"res_pred_r_{tag}"][:nshow])
        print("    +-            ", out[f"res_pred_r_err_{tag}"][:nshow])

        if order != 1:
            continue
        # 4. inhomogeneity of the residual variance.  Low-pass to the scales used by the
        # reconstruction (k < 0.4 h/Mpc) and compare the variance map with the predicted density.
        lowpass = (KMAG < 0.4) & VALID
        res_lp = np.fft.ifft2(res * lowpass).real * (fs.NG**2 / fs.LBOX)
        var_map = res_lp.var(axis=0, ddof=1)
        mu_lp = np.fft.ifft2(muk * lowpass).real * (fs.NG**2 / fs.LBOX)
        q = np.quantile(mu_lp, [0.25, 0.75])
        out["var_map"], out["mu_lp"] = var_map, mu_lp
        out["var_density_corr"] = np.corrcoef(var_map.ravel(), mu_lp.ravel())[0, 1]
        out["var_quartile_ratio"] = var_map[mu_lp > q[1]].mean() / var_map[mu_lp < q[0]].mean()
        # expected rms fluctuation of a variance estimate from 100 Gaussian samples: sqrt(2/99)
        out["var_map_cv"] = var_map.std() / var_map.mean()
        print(f"  variance map (k<0.4): corr with predicted density {out['var_density_corr']:.2f}, "
              f"top/bottom density quartile ratio {out['var_quartile_ratio']:.2f}, "
              f"rms/mean {out['var_map_cv']:.2f} (noise floor {np.sqrt(2 / (len(SIMS) - 1)):.2f})")
        # off-diagonal Fourier correlations between distinct modes, by bin pair
        half = lowpass & ((np.fft.fftfreq(fs.NG)[:, None] > 0) | ((np.fft.fftfreq(fs.NG)[:, None] == 0) & (np.fft.fftfreq(fs.NG)[None, :] > 0)))
        sel = np.flatnonzero(half.ravel())
        x = res.reshape(len(SIMS), -1)[:, sel]
        x = x - x.mean(axis=0)
        x = x / np.sqrt((np.abs(x) ** 2).mean(axis=0))
        corr = x.conj().T @ x / len(SIMS)                 # complex coherence matrix
        corr2 = x.T @ x / len(SIMS)                       # pseudo-coherence
        off = ~np.eye(len(sel), dtype=bool)
        null = 1.0 / len(SIMS) * (len(SIMS) / (len(SIMS) - 1))
        out["offdiag_mean_sq"] = float((np.abs(corr[off]) ** 2).mean())
        out["offdiag_pseudo_mean_sq"] = float((np.abs(corr2[off]) ** 2).mean())
        out["offdiag_null"] = null
        out["offdiag_nmodes"] = len(sel)
        kb = B_IDX[sel]
        m = np.zeros((nshow, nshow))
        for a in range(nshow):
            for b2 in range(nshow):
                blk = (np.abs(corr[np.ix_(kb == a, kb == b2)]) ** 2)
                if a == b2:
                    n = blk.shape[0]
                    m[a, b2] = (blk.sum() - np.trace(blk)) / max(n * (n - 1), 1)
                else:
                    m[a, b2] = blk.mean() if blk.size else np.nan
        out["offdiag_bins"] = m
        ex = out["offdiag_mean_sq"] - 1.0 / len(SIMS)
        print(f"  off-diagonal coherence of residual modes (k<0.4, {len(sel)} modes): <|rho|^2> = {out['offdiag_mean_sq']:.5f} "
              f"(null {1.0 / len(SIMS):.5f}; rms excess |rho| = {np.sqrt(max(ex, 0)):.3f}); pseudo {out['offdiag_pseudo_mean_sq']:.5f}")

    # ------------------------------------------------------------------ 5. Sigma^2 scan
    window = fs.load_window("holes")
    sig2_grid = np.linspace(6.0, 11.0, 501)
    ll = np.zeros((3, len(SIMS), len(sig2_grid)))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for oi, order in enumerate((0, 1, 2)):
        for j, s2v in enumerate(sig2_grid):
            post = fs.FieldPosterior(d_z, window, z, float(s2v), ptot, lpt_order=order, device=dev, dtype=torch.float64)
            if j == 0:
                zk = torch.fft.fft2(post.evolve(truth.to(dev)))
            pred = post.fft_norm(torch.fft.ifft2(zk * post.S).real * post.window)
            r = post.data_k - pred
            chi2 = ((r.real**2 + r.imag**2) / post.noise * post.valid).sum(dim=(-2, -1))
            # chi^2 is a sum over the observed area: fsky of the modes enter the normalisation
            ll[oi, :, j] = (-0.5 * (chi2 + post.fsky * (torch.log(post.noise) * post.valid).sum())).cpu().numpy()
    out.update(sig2_grid=sig2_grid, ll_scan=ll)
    for oi, name in enumerate(("EL", "1LPT", "2LPT")):
        tot = ll[oi].sum(axis=0)
        best = sig2_grid[np.argmax(tot)]
        inside = sig2_grid[tot > tot.max() - 0.5]
        print(f"Sigma^2 scan {name}: peak {best:.2f}  (1 sigma {inside.min():.2f}-{inside.max():.2f})")
    i82 = np.argmin(np.abs(sig2_grid - 8.2))
    print("at Sigma2=8.2: mean dlnL 1LPT-EL %.1f (min %.1f), 2LPT-1LPT %.2f +- %.2f, fraction 2LPT better %.2f" % (
        (ll[1] - ll[0])[:, i82].mean(), (ll[1] - ll[0])[:, i82].min(), (ll[2] - ll[1])[:, i82].mean(),
        (ll[2] - ll[1])[:, i82].std(), ((ll[2] - ll[1])[:, i82] > 0).mean()))

    fs.RESULTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(fs.RESULTS / "checks.npz", **out)
    print("saved checks.npz")


if __name__ == "__main__":
    main()
