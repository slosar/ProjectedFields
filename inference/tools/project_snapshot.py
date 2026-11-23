"""Project GADGET snapshots to 2D density fields along x, y and z and store them compactly.

Usage: python inference/tools/project_snapshot.py <snapshot.hdf5> <out.npz>
The projection is the one used by the likelihood code (CIC, 128^2, 250 Mpc/h).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from inference.projected_likelihood_torch import load_projected_field_from_snapshot  # noqa: E402

snap, out = sys.argv[1], sys.argv[2]
fields = {}
for axis in "xyz":
    f, z = load_projected_field_from_snapshot(snap, ngrid=128, lbox=250.0, axis=axis)
    fields[axis] = f.numpy().astype(np.float32)
np.savez_compressed(out, redshift=z, **fields)
print(out, "z=%.4f" % z, "rms", " ".join("%.4f" % fields[a].std() for a in "xyz"))
