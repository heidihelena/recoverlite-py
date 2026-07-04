"""Generate varied LMM validation datasets as CSVs."""
import json
import os
import sys

import numpy as np

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(20260704)
configs = [
    dict(name="balanced_16x30_icc05", G=16, m=[30] * 16, icc=0.05, tau=0.4),
    dict(name="balanced_16x30_icc15", G=16, m=[30] * 16, icc=0.15, tau=0.4),
    dict(name="small_8x10_icc05", G=8, m=[10] * 8, icc=0.05, tau=0.4),
    dict(name="unbalanced_12", G=12, m=list(rng.integers(5, 40, 12)),
         icc=0.10, tau=0.3),
    dict(name="near_zero_icc", G=16, m=[30] * 16, icc=0.005, tau=0.4),
    dict(name="null_effect", G=16, m=[30] * 16, icc=0.05, tau=0.0),
]
for cfg in configs:
    G, sizes, icc, tau = cfg["G"], cfg["m"], cfg["icc"], cfg["tau"]
    z_cl = np.zeros(G, dtype=int)
    z_cl[rng.permutation(G)[: G // 2]] = 1
    rows = []
    for j in range(G):
        u = rng.normal(0, np.sqrt(icc))
        for _ in range(int(sizes[j])):
            e = rng.normal(0, np.sqrt(1 - icc))
            y = tau * z_cl[j] + u + e
            rows.append((y, z_cl[j], j))
    with open(os.path.join(out, cfg["name"] + ".csv"), "w") as fh:
        fh.write("y,treatment,cluster\n")
        for y, z, c in rows:
            fh.write(f"{y:.12g},{z},{c}\n")
print(json.dumps([c["name"] for c in configs]))
