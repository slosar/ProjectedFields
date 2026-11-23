"""Figures of Section 4 (`Likelihood and sampling') of the paper.

Reads $PROFIELD_DATA/results/*.npz (made by model_checks.py and run_ensemble.py; see
inference/run_all.sh) and writes PDFs into figures/.  Also prints the numbers
quoted in the text.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference import field_sampler as fs  # noqa: E402

ROOT = fs.ROOT
RES = fs.RESULTS
OUT = ROOT / "figures"
L = fs.LBOX
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "font.size": 11})
np.set_printoptions(precision=3, suppress=True, linewidth=200)
KMAX_PLOT = 0.55


SHARED = {"k_edges", "window", "z", "sigma2", "lpt", "axis", "mask", "ptot_kind", "mask_noise", "n_burn",
          "n_keep", "num_steps", "k_mean", "n_modes", "k_shell"}


def load(tag: str):
    """Load a run; `a+b` concatenates the realizations of runs a and b."""
    parts = [np.load(RES / f"{t}.npz") for t in tag.split("+")]
    if len(parts) == 1:
        return parts[0]
    return {k: (parts[0][k] if k in SHARED else np.concatenate([q[k] for q in parts], axis=0)) for k in parts[0].files}


def mask_shapes():
    wf = np.load(ROOT / "inference" / "window_holes.npz")
    return wf["window"], wf.get("circles"), wf.get("rects")


def fig_mask():
    w, _, _ = mask_shapes()
    fig, ax = plt.subplots(figsize=(4.6, 4.4), constrained_layout=True)
    ax.imshow(w, origin="lower", cmap="gray", vmin=0, vmax=1, extent=[0, L, 0, L])
    ax.set_xlabel(r"$h^{-1}$ Mpc"); ax.set_ylabel(r"$h^{-1}$ Mpc")
    ax.set_title(f"window ($f_\\mathrm{{sky}}={float((w**2).mean()):.2f}$)")
    fig.savefig(OUT / "like_mask.pdf"); plt.close(fig)


def draw_mask(ax, circles, rects, color="lime", lw=1.4):
    if circles is not None:
        for cx, cy, r in circles:
            ax.add_patch(mpatches.Circle((cx, cy), r, fill=False, edgecolor=color, lw=lw))
    if rects is not None:
        for x0, y0, w, h in rects:
            ax.add_patch(mpatches.Rectangle((x0, y0), w, h, fill=False, edgecolor=color, lw=lw))
    ax.set_xticks([]); ax.set_yticks([])


def damp(field: np.ndarray, sigma2: float) -> np.ndarray:
    k = fs.kgrid()[2]
    return np.fft.ifft2(np.fft.fft2(field) * np.exp(-0.5 * k**2 * sigma2)).real


# --------------------------------------------------------------------- fixed initial conditions
def fig_sigma2_posterior(c):
    ll, grid = c["ll_scan"], c["sig2_grid"]
    labels = {0: "EL", 1: "1LPT", 2: "2LPT"}; colors = {0: "tab:green", 1: "tab:red", 2: "tab:blue"}
    fig, ax = plt.subplots(figsize=(6.0, 4.2), constrained_layout=True)
    for oi in (0, 1, 2):
        tot = ll[oi].sum(axis=0)
        ax.plot(grid, np.exp(tot - tot.max()), color=colors[oi], lw=2,
                label=f"{labels[oi]} ($\\Sigma^2$={grid[np.argmax(tot)]:.2f})")
    ax.set_xlim(7.5, 10.5); ax.set_xlabel(r"$\Sigma^2\ [(\mathrm{Mpc}/h)^2]$")
    ax.set_ylabel(r"$P(\Sigma^2)$ (peak-normalised)")
    ax.legend(frameon=False)
    fig.savefig(OUT / "like_sigma2_posterior.pdf"); plt.close(fig)


def fig_order_comparison(c):
    ll, grid = c["ll_scan"], c["sig2_grid"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True, sharex=True)
    for a, d, ttl, col in [(ax[0], ll[1] - ll[0], r"$\ln\mathcal{L}_\mathrm{1LPT}-\ln\mathcal{L}_\mathrm{EL}$", "tab:green"),
                           (ax[1], ll[2] - ll[1], r"$\ln\mathcal{L}_\mathrm{2LPT}-\ln\mathcal{L}_\mathrm{1LPT}$", "tab:blue")]:
        for i in range(d.shape[0]):
            a.plot(grid, d[i], color=col, alpha=0.12, lw=0.7)
        a.plot(grid, d.mean(axis=0), color="k", lw=2.2, label="mean")
        a.axhline(0, color="r", ls=":", lw=1.0); a.set_xlim(grid.min(), grid.max())
        a.set_xlabel(r"$\Sigma^2\ [(\mathrm{Mpc}/h)^2]$"); a.set_ylabel(ttl); a.legend(frameon=False)
    fig.savefig(OUT / "like_order_comparison.pdf"); plt.close(fig)


def fig_residual(c):
    k = c["k_mean"]; m = k < KMAX_PLOT
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0), constrained_layout=True)
    for tag, col, lab in (("lpt0", "tab:green", "EL"), ("lpt1", "tab:red", "1LPT"), ("lpt2", "tab:blue", "2LPT")):
        ax[0].semilogx(k[m], c[f"res_over_noise_{tag}"][m], color=col, lw=2, label=lab)
        ax[1].errorbar(k[m], c[f"res_pred_r_{tag}"][m], yerr=c[f"res_pred_r_err_{tag}"][m], color=col, lw=1.5, marker="o", ms=3, label=lab)
    ax[0].axhline(1, color="k", ls=":", lw=0.8); ax[0].set_ylim(0, 1.5)
    ax[0].set_ylabel(r"$\langle|\Delta_{\rm obs}-S\mathcal{Z}|^2\rangle\,/\,(1-S^2)P_{\rm tot}$")
    ax[1].axhline(0, color="k", ls=":", lw=0.8); ax[1].set_xscale("log"); ax[1].set_ylim(-0.5, 0.5)
    ax[1].set_ylabel(r"correlation of $\Delta_{\rm obs}-S\mathcal{Z}$ with $S\mathcal{Z}$")
    for a in ax:
        a.set_xlabel(r"$k_\perp\ [h\,\mathrm{Mpc}^{-1}]$"); a.legend(frameon=False)
    fig.savefig(OUT / "like_residual.pdf"); plt.close(fig)


# ------------------------------------------------------------------------- single realization
def fig_recovery(d, chain=0):
    w, circ, rect = mask_shapes()
    s2 = float(d["sigma2"])
    td = d["true_damped"][chain]; mean = d["mean_damped"][chain]
    sample = damp(d["samples"][chain, -1], s2)
    data = d["data"][chain]; pred = d["mean_pred"][chain]
    v = float(np.abs(td).max()); kw = dict(origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
    vd = float(np.abs(data).max()); kwd = dict(origin="lower", cmap="RdBu_r", vmin=-vd, vmax=vd)
    fig, ax = plt.subplots(2, 3, figsize=(11, 7.4), constrained_layout=True)
    ax[0, 0].imshow(td, **kw); ax[0, 0].set_title("true IC (damped)")
    ax[0, 1].imshow(sample, **kw); ax[0, 1].set_title("posterior sample")
    ax[0, 2].imshow(mean, **kw); ax[0, 2].set_title("posterior mean")
    ax[1, 0].imshow(data, **kwd); ax[1, 0].set_title("projected data")
    ax[1, 1].imshow(pred, **kwd); ax[1, 1].set_title("mean recovered prediction")
    ax[1, 2].imshow(data - pred, **kwd); ax[1, 2].set_title("data $-$ mean prediction")
    for a in ax.ravel():
        draw_mask(a, circ, rect)
    fig.savefig(OUT / "like_recovery.pdf"); plt.close(fig)


def fig_std(d):
    w, circ, rect = mask_shapes(); hole = w < 0.5
    # combine the chains: total variance = mean within-chain variance + variance of chain means
    def comb(mean, std):
        return np.sqrt((std**2).mean(axis=0) + mean.var(axis=0))
    ic_std = comb(d["mean_damped"], d["std_damped"]); pr_std = comb(d["mean_pred"], d["std_pred"])
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.3), constrained_layout=True)
    for a, s, ttl in [(ax[0], ic_std, "IC posterior std (damped)"), (ax[1], pr_std, "prediction posterior std")]:
        im = a.imshow(s, origin="lower", cmap="viridis"); a.set_title(ttl)
        fig.colorbar(im, ax=a, fraction=0.046); draw_mask(a, circ, rect)
    fig.savefig(OUT / "like_std.pdf"); plt.close(fig)
    print(f"std maps: prediction std inside/outside mask = {pr_std[hole].mean() / pr_std[~hole].mean():.2f}, "
          f"IC (damped) = {ic_std[hole].mean() / ic_std[~hole].mean():.2f}")


def sample_r(d):
    return d["p_st_iter"] / np.sqrt(d["p_ss_iter"] * d["p_tt"][:, None, :])


def mean_r(d):
    return d["p_mt"] / np.sqrt(d["p_mm"] * d["p_tt"])


def fig_xcorr(d1, d2):
    k = d1["k_mean"]
    fig, ax = plt.subplots(figsize=(6.0, 4.4), constrained_layout=True)
    for d, col, lab, off in [(d1, "tab:red", "1LPT", 0.985), (d2, "tab:blue", "2LPT", 1.015)]:
        rs = sample_r(d).reshape(-1, len(k))
        lo, mid, hi = np.percentile(rs, [16, 50, 84], axis=0)
        ax.fill_between(k, lo, hi, color=col, alpha=0.25)
        ax.semilogx(k, mid, color=col, lw=2, label=f"{lab}, posterior samples")
        # posterior mean of all chains combined
        mm = np.fft.fft2(d["mean_field"].mean(axis=0)); tt = np.fft.fft2(d["true_field"][0])
        b, ok = fs.bin_modes(d["k_edges"]); nb = len(k)
        bn = lambda x: np.bincount(b[ok], weights=x.ravel()[ok], minlength=nb)  # noqa: E731
        r = bn((mm * tt.conj()).real) / np.sqrt(bn(np.abs(mm) ** 2) * bn(np.abs(tt) ** 2))
        ax.semilogx(k * off, r, color=col, ls="none", marker="o", ms=4, label=f"{lab}, posterior mean")
    ax.axhline(1, color="k", ls=":", lw=0.8); ax.axhline(0, color="k", lw=0.6); ax.set_ylim(-0.2, 1.1)
    ax.set_xlabel(r"$k_\perp\ [h\,\mathrm{Mpc}^{-1}]$")
    ax.set_ylabel(r"$r(k_\perp)$ with true initial field")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.savefig(OUT / "like_xcorr.pdf"); plt.close(fig)


# ----------------------------------------------------------------------------------- ensemble
def boot(fn, n, nboot=300, seed=0):
    rng = np.random.default_rng(seed)
    return np.std([fn(rng.integers(0, n, n)) for _ in range(nboot)], axis=0)


def coverage(d, lo, hi):
    """Fraction of Fourier modes (per k-bin) and of pixels of the damped field (observed /
    masked area) whose truth lies inside the central posterior interval [lo, hi]."""
    b, ok = fs.bin_modes(d["k_edges"]); nb = len(d["k_mean"])
    ins = lambda r: ((r > lo) & (r < hi)).astype(float)  # noqa: E731
    f = 0.5 * (ins(d["rank_re"]) + ins(d["rank_im"])).reshape(len(d["rank_re"]), -1)
    per_bin = np.array([f[:, ok & (b == i)].mean() for i in range(nb)])
    obs = d["window"] > 0.5
    pix = ins(d["rank_damped"])
    return per_bin, pix[:, obs].mean(), (pix[:, ~obs].mean() if (~obs).any() else np.nan)


def ensemble_stats(d):
    n = len(d["sims"])
    idx = np.arange(n)
    slope = lambda i: d["p_mt"][i].mean(axis=0) / d["p_mm"][i].mean(axis=0)              # noqa: E731
    transfer = lambda i: d["p_mt"][i].mean(axis=0) / d["p_tt"][i].mean(axis=0)           # noqa: E731
    samp = lambda i: np.sqrt(d["p_ss"][i].mean(axis=0) / d["p_tt"][i].mean(axis=0))      # noqa: E731
    r = mean_r(d)
    return dict(k=d["k_mean"], r_mean=r.mean(axis=0), r_std=r.std(axis=0),
                slope=slope(idx), slope_err=boot(slope, n), transfer=transfer(idx), transfer_err=boot(transfer, n),
                samp=samp(idx), samp_err=boot(samp, n),
                chi2=d["chi2_iter"].mean(axis=(0, 1)), accept=d["accept"])


def fig_ensemble(runs: dict):
    """runs: label -> (npz, color).  All runs must have independent Delta_0 (the amplitude and
    confidence-limit tests are only meaningful on average over Delta_0)."""
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.0), constrained_layout=True, sharex=True)
    for lab, (d, col) in runs.items():
        st = ensemble_stats(d); k = st["k"]; m = k < KMAX_PLOT
        ax[0, 0].fill_between(k[m], (st["r_mean"] - st["r_std"])[m], (st["r_mean"] + st["r_std"])[m], color=col, alpha=0.15)
        ax[0, 0].plot(k[m], st["r_mean"][m], color=col, lw=2, label=lab)
        ax[0, 1].errorbar(k[m], st["slope"][m], yerr=st["slope_err"][m], color=col, lw=1.5, marker="o", ms=3, label=lab)
        ax[1, 0].errorbar(k[m], st["samp"][m], yerr=st["samp_err"][m], color=col, lw=1.5, marker="o", ms=3, label=lab)
        for (lo, hi), ls in (((0.16, 0.84), "-"), ((0.025, 0.975), "--")):
            cb, _, _ = coverage(d, lo, hi)
            ax[1, 1].plot(k[m], cb[m], color=col, ls=ls, lw=1.8, marker="o", ms=3)
    ax[0, 0].set_ylabel(r"$r(k_\perp)$ of posterior mean with truth"); ax[0, 0].set_ylim(0, 1.05)
    ax[0, 1].set_ylabel(r"$P_{\rm mean\times true}/P_{\rm mean\times mean}$"); ax[0, 1].set_ylim(0.7, 1.5)
    ax[1, 0].set_ylabel(r"$\sqrt{P_{\rm sample}/P_{\rm true}}$"); ax[1, 0].set_ylim(0.7, 1.5)
    ax[1, 1].set_ylabel("fraction within 68% and 95% confidence lim."); ax[1, 1].set_ylim(0.5, 1.02)
    for a in (ax[0, 1], ax[1, 0]):
        a.axhline(1, color="k", ls=":", lw=0.8)
    ax[1, 1].axhline(0.68, color="k", ls=":", lw=0.8); ax[1, 1].axhline(0.95, color="k", ls=":", lw=0.8)
    for a in ax.ravel():
        a.set_xscale("log")
    for a in ax[1]:
        a.set_xlabel(r"$k_\perp\ [h\,\mathrm{Mpc}^{-1}]$")
    ax[0, 0].legend(frameon=False, fontsize=10, loc="lower left")
    ax[1, 1].legend(handles=[Line2D([], [], color="k", ls="-", lw=1.8, label="68%"),
                             Line2D([], [], color="k", ls="--", lw=1.8, label="95%")],
                    frameon=False, fontsize=10, loc="lower left")
    fig.savefig(OUT / "like_ensemble.pdf"); plt.close(fig)


def print_run(tag, d):
    st = ensemble_stats(d); n = 11
    print(f"=== {tag}: {len(d['sims'])} chains, accept {st['accept'].mean():.2f} (min {st['accept'].min():.2f}), mask weighting {d['mask_noise'] if 'mask_noise' in d else 'fsky'}")
    print("  k            ", st["k"][:n])
    print("  r(mean)      ", st["r_mean"][:n]); print("    scatter    ", st["r_std"][:n])
    print("  slope mt/mm  ", st["slope"][:n]); print("    +-         ", st["slope_err"][:n])
    print("  transfer mt/tt", st["transfer"][:n])
    print("  sqrt(Pss/Ptt)", st["samp"][:n]); print("    +-         ", st["samp_err"][:n])
    print("  chi2 per mode", st["chi2"][:n])
    for name, (lo, hi) in (("68%", (0.16, 0.84)), ("95%", (0.025, 0.975))):
        cb, obs, hole = coverage(d, lo, hi)
        print(f"  coverage {name}: per bin", cb[:n], f" damped field: observed {obs:.3f}, masked {hole:.3f}")
    from inference.field_sampler import ess
    e_u = np.array([ess(u) for u in d["potential"]])
    e_t = np.array([[ess(d["track"][c, :, j]) for j in range(d["track"].shape[2])] for c in range(len(d["track"]))])
    print(f"  ESS of {d['potential'].shape[1]} kept iterations: potential median {np.median(e_u):.0f} (min {e_u.min():.0f}); "
          f"low-k modes median {np.median(e_t):.0f} (min {e_t.min():.0f})")
    if len(set(d["sims"].tolist())) == 1 and len(d["sims"]) > 1:
        x = d["track"]; nn = x.shape[1]
        wv = x.var(axis=1, ddof=1).mean(axis=0); bv = nn * x.mean(axis=1).var(axis=0, ddof=1)
        print("  R-hat (tracked low-k modes): max %.3f" % np.sqrt(((nn - 1) / nn * wv + bv / nn) / wv).max())
        mdm = d["mean_damped"]
        print("  chain-to-chain rms of posterior mean / posterior std: %.3f" % (mdm.std(axis=0) / d["std_damped"].mean(axis=0)).mean())


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_mask()
    have = {p.stem for p in RES.glob("*.npz")}
    if "checks" in have:
        c = load("checks")
        fig_sigma2_posterior(c); fig_order_comparison(c); fig_residual(c)
    for tag in sorted(have - {"checks", "free_ptot"}):
        print_run(tag, load(tag))
    if {"r88_lpt1", "r88_lpt2"} <= have:
        fig_recovery(load("r88_lpt1")); fig_std(load("r88_lpt1")); fig_xcorr(load("r88_lpt1"), load("r88_lpt2"))
    indep = "x_mask_lpt1+y_mask_lpt1" if {"x_mask_lpt1", "y_mask_lpt1"} <= have else "x_mask_lpt1"
    indep_nm = "x_nomask_lpt1+y_nomask_lpt1" if {"x_nomask_lpt1", "y_nomask_lpt1"} <= have else "x_nomask_lpt1"
    runs = {}
    for tag, lab, col in ((indep, "with mask", "tab:blue"), (indep_nm, "without mask", "tab:red")):
        if set(tag.split("+")) <= have:
            runs[lab] = (load(tag), col)
            if "+" in tag:
                print_run(tag, load(tag))
    for tag in ("x_mask_lpt2+y_mask_lpt2", "x_mask_lpt1_fsky+y_mask_lpt1_fsky"):
        if set(tag.split("+")) <= have:
            print_run(tag, load(tag))
    if runs:
        fig_ensemble(runs)
    print("DONE")
