"""Sample the initial-field posterior for many realizations (or many chains of one realization)
and store calibration summaries.

Examples (from the repository root):

    # 100 realizations sharing the projected modes (z projection), masked, 1LPT, suite P_tot
    python inference/run_ensemble.py --axis z --sims 0-99 --tag z_mask_lpt1
    # 100 realizations with independent projected modes (x projection)
    python inference/run_ensemble.py --axis x --sims 0-99 --tag x_mask_lpt1
    # 8 independent chains of realization 88, keeping posterior samples for the figures
    python inference/run_ensemble.py --axis z --sims 88 --chains 8 --n-keep 1500 --save-samples 32 --tag r88_lpt1

Output: $PROFIELD_DATA/results/<tag>.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference import field_sampler as fs  # noqa: E402

OUT = fs.RESULTS
# pixels tracked for convergence diagnostics: two inside mask holes, two outside
TRACK_MODES = [(0, 1), (1, 0), (1, 1), (0, 2), (2, 1), (0, 4), (3, 3), (0, 8)]


def parse_sims(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


class Accumulator:
    """Running posterior summaries for one batch."""

    def __init__(self, post: fs.FieldPosterior, truth: np.ndarray, k_edges: np.ndarray, n_keep: int,
                 save_samples: int) -> None:
        self.post, dev, dt = post, post.device, post.dtype
        B = post.B
        self.n = 0
        b, ok = fs.bin_modes(k_edges)
        nb = len(k_edges) - 1
        onehot = np.zeros((fs.NG**2, nb))
        onehot[np.arange(fs.NG**2)[ok], b[ok]] = 1.0
        self.counts = onehot.sum(axis=0)
        self.binmat = torch.as_tensor(onehot / np.maximum(self.counts, 1), device=dev, dtype=dt)
        # truth: deconvolve the CIC window of the particle deposit, keep the modes the model has
        tk = np.fft.fft2(truth) / fs.cic_window()
        self.t_modes = torch.as_tensor(tk, device=dev, dtype=post.cdtype) * post.valid
        self.t_field = torch.fft.ifft2(self.t_modes).real
        self.t_damped = torch.fft.ifft2(self.t_modes * post.S).real
        z = lambda: torch.zeros(B, fs.NG, fs.NG, device=dev, dtype=torch.float64)  # noqa: E731
        self.s_f, self.s_f2, self.s_fd, self.s_fd2, self.s_mf, self.s_mf2 = z(), z(), z(), z(), z(), z()
        self.rank_fd, self.rank_re, self.rank_im, self.s_m2 = z(), z(), z(), z()
        self.pss = np.zeros((n_keep, B, nb), dtype=np.float32)
        self.pst = np.zeros((n_keep, B, nb), dtype=np.float32)
        self.chi2 = np.zeros((n_keep, B, nb), dtype=np.float32)
        self.track = np.zeros((n_keep, B, len(TRACK_MODES)), dtype=np.float32)
        self.save_every = max(1, n_keep // save_samples) if save_samples else 0
        self.samples: list[np.ndarray] = []

    def binned(self, per_mode: torch.Tensor) -> torch.Tensor:
        return per_mode.reshape(per_mode.shape[0], -1).to(self.binmat.dtype) @ self.binmat

    @torch.no_grad()
    def __call__(self, it: int, xi: torch.Tensor) -> None:
        post = self.post
        m = post.modes(xi)
        f = torch.fft.ifft2(m).real
        fd = torch.fft.ifft2(m * post.S).real
        mf = post.mean_field(f)
        self.s_f += f; self.s_f2 += f**2
        self.s_fd += fd; self.s_fd2 += fd**2
        self.s_mf += mf; self.s_mf2 += mf**2
        self.rank_fd += fd < self.t_damped
        self.rank_re += m.real < self.t_modes.real
        self.rank_im += m.imag < self.t_modes.imag
        m2 = m.real**2 + m.imag**2
        self.s_m2 += m2
        self.pss[it] = self.binned(m2).cpu().numpy()
        self.pst[it] = self.binned((m * self.t_modes.conj()).real).cpu().numpy()
        r = post.data_k - post.fft_norm(mf * post.window)
        self.chi2[it] = self.binned((r.real**2 + r.imag**2) / post.noise * post.valid).cpu().numpy()
        for j, (a, b) in enumerate(TRACK_MODES):
            self.track[it, :, j] = m[:, a, b].real.cpu().numpy()
        if self.save_every and it % self.save_every == 0:
            self.samples.append(f.cpu().numpy().astype(np.float32))
        self.n += 1

    def finish(self) -> dict:
        n = self.n
        c = lambda t: t.cpu().numpy()  # noqa: E731
        mean_f, mean_fd, mean_mf = self.s_f / n, self.s_fd / n, self.s_mf / n
        mm = torch.fft.fft2(mean_f.to(self.post.dtype))
        tt = self.t_modes
        out = {
            "mean_field": c(mean_f).astype(np.float32),
            "std_field": c((self.s_f2 / n - mean_f**2).clamp_min(0).sqrt()).astype(np.float32),
            "mean_damped": c(mean_fd).astype(np.float32),
            "std_damped": c((self.s_fd2 / n - mean_fd**2).clamp_min(0).sqrt()).astype(np.float32),
            "mean_pred": c(mean_mf).astype(np.float32),
            "std_pred": c((self.s_mf2 / n - mean_mf**2).clamp_min(0).sqrt()).astype(np.float32),
            "rank_damped": c(self.rank_fd / n).astype(np.float32),
            "rank_re": c(self.rank_re / n).astype(np.float32),
            "rank_im": c(self.rank_im / n).astype(np.float32),
            "true_field": c(self.t_field).astype(np.float32),
            "true_damped": c(self.t_damped).astype(np.float32),
            "p_mm": c(self.binned(mm.real**2 + mm.imag**2)),
            "p_mt": c(self.binned((mm * tt.conj()).real)),
            "p_tt": c(self.binned(tt.real**2 + tt.imag**2)),
            "p_ss": c(self.binned(self.s_m2 / n)),
            "p_ss_iter": self.pss.transpose(1, 0, 2),      # (B, n_keep, nb)
            "p_st_iter": self.pst.transpose(1, 0, 2),
            "chi2_iter": self.chi2.transpose(1, 0, 2),
            "track": self.track.transpose(1, 0, 2),
        }
        if self.samples:
            out["samples"] = np.stack(self.samples, axis=1)  # (B, n_samples, N, N)
        return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--axis", default="z", choices=["x", "y", "z"])
    p.add_argument("--sims", default="0-99")
    p.add_argument("--chains", type=int, default=1, help="independent chains per realization")
    p.add_argument("--mask", default="holes", choices=["holes", "none"])
    p.add_argument("--lpt", type=int, default=1, choices=[0, 1, 2])
    p.add_argument("--ptot", default="suite", choices=["suite", "data"])
    p.add_argument("--sigma2", type=float, default=8.2)
    p.add_argument("--n-burn", type=int, default=300)
    p.add_argument("--n-keep", type=int, default=600)
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--batch", type=int, default=50)
    p.add_argument("--nbins", type=int, default=15)
    p.add_argument("--save-samples", type=int, default=0)
    p.add_argument("--init", default="zero", choices=["zero", "prior", "truth"])
    p.add_argument("--damp-after-mask", action="store_true", help="old ordering of mask and damping")
    p.add_argument("--mask-noise", default="plain", choices=["plain", "fsky", "convolved"],
                   help="variance assigned to the masked residual modes (see FieldPosterior)")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", required=True)
    args = p.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    sims = [s for s in parse_sims(args.sims) for _ in range(args.chains)]
    k_edges = fs.default_k_edges(args.nbins)
    window = fs.load_window(args.mask)
    data, z = fs.load_evolved(sims, args.axis)
    truth = fs.load_true_initial(sims, args.axis)
    print(f"{args.tag}: {len(sims)} chains, axis={args.axis} mask={args.mask} lpt={args.lpt} ptot={args.ptot} "
          f"z={z:.3f} device={device}", flush=True)

    results: dict[str, list] = {}
    t0 = time.time()
    for start in range(0, len(sims), args.batch):
        sl = slice(start, start + args.batch)
        chunk = sims[sl]
        if args.ptot == "suite":
            ptot = np.stack([fs.suite_ptot(exclude=None if args.axis == "z" else i) for i in chunk])
        else:
            ptot = np.stack([fs.data_ptot(d, window, k_edges) for d in data[sl]])
        post = fs.FieldPosterior(data[sl], window, z, args.sigma2, ptot, lpt_order=args.lpt,
                                 damp_before_mask=not args.damp_after_mask, mask_noise=args.mask_noise,
                                 device=device)
        acc = Accumulator(post, truth[sl], k_edges, args.n_keep, args.save_samples)
        xi0 = None
        if args.init == "truth":
            xi0 = post.xi_from_field(acc.t_field)
        elif args.init == "prior":
            g = torch.Generator(device=post.device).manual_seed(1000 + args.seed + start)
            xi0 = torch.randn(post.B, fs.NG, fs.NG, device=post.device, dtype=post.dtype, generator=g)
        run = fs.hmc(post, n_burn=args.n_burn, n_keep=args.n_keep, num_steps=args.num_steps,
                     seed=args.seed + start, xi_init=xi0, callback=acc)
        res = acc.finish()
        res.update(accept=run["accept"], step=run["step"], potential=run["potential"].T,
                   loglike_truth=post.log_like(acc.t_field).cpu().numpy(),
                   ptot_shell=np.stack([fs.shell_average(pt)[1] for pt in ptot]))
        for k, v in res.items():
            results.setdefault(k, []).append(v)
        print(f"  chains {start}-{start + len(chunk) - 1} done, accept={run['accept'].mean():.2f} "
              f"step={run['step'].mean():.4f}  elapsed {time.time() - t0:.0f}s", flush=True)

    out = {k: np.concatenate(v, axis=0) for k, v in results.items()}
    b, ok = fs.bin_modes(k_edges)
    kmag = fs.kgrid()[2].ravel()
    nb = len(k_edges) - 1
    out.update(
        sims=np.array(sims), k_edges=k_edges, window=window, z=z, sigma2=args.sigma2, lpt=args.lpt,
        axis=args.axis, mask=args.mask, ptot_kind=args.ptot, mask_noise=args.mask_noise,
        n_burn=args.n_burn, n_keep=args.n_keep,
        num_steps=args.num_steps,
        k_mean=np.bincount(b[ok], weights=kmag[ok], minlength=nb) / np.maximum(np.bincount(b[ok], minlength=nb), 1),
        n_modes=np.bincount(b[ok], minlength=nb), data=data.astype(np.float32),
        k_shell=fs.shell_average(np.zeros((fs.NG, fs.NG)))[0],
    )
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{args.tag}.npz", **out)
    print(f"saved {OUT / (args.tag + '.npz')}  total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
