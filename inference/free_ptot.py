"""Why P_tot cannot be a free parameter constrained only through the residual.

Samples, for one realization, the posterior with P_tot free,

    P(Delta_0, P_tot | Delta_obs)  with  pi(P_tot) ~ 1/P_tot  per k-bin,

by HMC on the field alternated with the exact inverse-gamma draw of each P_tot bin, and
compares with (a) P_tot fixed to the suite mean (the likelihood of the paper) and (b) the
likelihood of the previous version, where an extra `measured power' term anchored P_tot.
No mask is used, so that the posterior is exactly the one above.  Two priors are tried,
pi ~ 1/P_tot ("free_jeffreys") and pi = const ("free_flat").

Per bin with n modes the conditional density of P is P^(-n/2) exp(-A/2P) pi(P), with
A = sum |residual|^2 / (1 - S^2).  The field can make A vanish in a bin at a finite prior cost,
and the density marginalised over P behaves as A^(-n/2) (Jeffreys) or A^(-n/2+1) (flat) there:
for the scale-invariant prior the posterior is improper.

    python inference/free_ptot.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference import field_sampler as fs  # noqa: E402

np.set_printoptions(precision=2, suppress=True, linewidth=200)
SIM, CHAINS, NSHOW = 88, 4, 11
k_edges = fs.default_k_edges(15)
b_idx, b_ok = fs.bin_modes(k_edges)
nb = len(k_edges) - 1
window = fs.load_window("none")
data, z = fs.load_evolved([SIM] * CHAINS, "z")
truth = fs.load_true_initial([SIM], "z")[0]
dev = "cuda" if torch.cuda.is_available() else "cpu"
ptot_suite = fs.suite_ptot()
kmag = fs.kgrid()[2].ravel()
k_mean = np.bincount(b_idx[b_ok], weights=kmag[b_ok], minlength=nb) / np.bincount(b_idx[b_ok], minlength=nb)
print("k            ", k_mean[:NSHOW])

results = {}
for mode in ("fixed", "free_jeffreys", "free_flat", "anchored"):
    post = fs.FieldPosterior(data, window, z, 8.2, ptot_suite, device=dev)
    onehot = torch.zeros(fs.NG**2, nb, device=dev)
    onehot[torch.as_tensor(np.arange(fs.NG**2)[b_ok], device=dev), torch.as_tensor(b_idx[b_ok], device=dev)] = 1.0
    counts = onehot.sum(dim=0)
    bin_of_mode = torch.as_tensor(np.clip(b_idx, 0, nb - 1), device=dev)
    resid_frac = (1.0 - post.S**2).clamp_min(1e-12)        # N = (1 - S^2) P_tot
    # measured (mask-deconvolved) binned data power, the anchor of the previous version
    phat = ((post.data_k.abs() ** 2).reshape(CHAINS, -1) @ onehot) / counts / post.fsky
    suite_b = (torch.as_tensor(ptot_suite, device=dev, dtype=torch.float32).reshape(1, -1) @ onehot) / counts
    ptot_hist = []

    def gibbs(it, xi, post=post, mode=mode):
        with torch.no_grad():
            r = post.data_k - post.predicted_k(post.initial_field(xi))
            a = (((r.real**2 + r.imag**2) / resid_frac * post.valid).reshape(CHAINS, -1) @ onehot)
            if mode == "free_jeffreys":   # residual term only, pi ~ 1/P
                alpha, beta = counts / 2.0, a / 2.0
            elif mode == "free_flat":     # residual term only, pi = const
                alpha, beta = (counts / 2.0 - 1.0).clamp_min(0.5), a / 2.0
            else:                       # previous version: residual + measured-power term
                alpha, beta = counts - 1.0, (a + counts * phat) / 2.0
            g = torch.distributions.Gamma(alpha.expand(CHAINS, nb), torch.ones(CHAINS, nb, device=dev)).sample()
            pb = (beta / g.clamp_min(1e-20)).clamp_min(1e-12)
            post.noise = (resid_frac * pb[:, bin_of_mode].reshape(CHAINS, fs.NG, fs.NG)).clamp_min(1e-12)
            ptot_hist.append((pb / suite_b).cpu().numpy())

    tk = torch.as_tensor(np.fft.fft2(truth) / fs.cic_window(), device=dev, dtype=torch.complex64) * post.valid
    stats = {"pss": [], "pst": [], "sum": 0.0}

    def cb(it, xi, post=post, stats=stats):
        m = post.modes(xi)
        stats["pss"].append((((m.abs() ** 2).reshape(CHAINS, -1) @ onehot) / counts).cpu().numpy())
        stats["pst"].append((((m * tk.conj()).real.reshape(CHAINS, -1) @ onehot) / counts).cpu().numpy())
        stats["sum"] = stats["sum"] + m

    run = fs.hmc(post, n_burn=300, n_keep=600, num_steps=50, seed=1, callback=cb,
                 update=None if mode == "fixed" else gibbs, verbose=False)
    ptt = (((tk.abs() ** 2).reshape(1, -1) @ onehot) / counts).cpu().numpy()
    mm = stats["sum"] / len(stats["pss"])
    r_mean = ((((mm * tk.conj()).real.reshape(CHAINS, -1) @ onehot) / counts).cpu().numpy()
              / np.sqrt((((mm.abs() ** 2).reshape(CHAINS, -1) @ onehot) / counts).cpu().numpy() * ptt))
    r_samp = np.array(stats["pst"]) / np.sqrt(np.array(stats["pss"]) * ptt)
    print(f"--- P_tot {mode}: acceptance {run['accept'].mean():.2f}")
    if ptot_hist:
        ph = np.array(ptot_hist[300:])
        print("  median P_tot/P_suite", np.median(ph, axis=(0, 1))[:NSHOW])
        print("  min    P_tot/P_suite", ph.min(axis=(0, 1))[:NSHOW])
    print("  r(k) posterior mean ", r_mean.mean(axis=0)[:NSHOW])
    print("  r(k) single samples ", np.median(r_samp, axis=(0, 1))[:NSHOW])
    results[mode] = dict(accept=run["accept"], r_mean=r_mean, r_samp=np.median(r_samp, axis=(0, 1)),
                         ptot=np.array(ptot_hist[300:]) if ptot_hist else np.zeros(0))
out = fs.RESULTS
out.mkdir(parents=True, exist_ok=True)
np.savez(out / "free_ptot.npz", k_mean=k_mean, **{f"{m}_{k}": v for m, d in results.items() for k, v in d.items()})
