"""Second pass over the range scenarios: more centre offsets around each scenario's best separation,
then the best centre per range compared with the geometric-mean rule.

The rule puts the pair's dioptric centre at 1/sqrt(z_lo * z_hi), i.e. an offset
delta = -(sqrt(rho) - 1) / (sqrt(rho) + 1) with rho = z_hi / z_lo.

Usage: python dfdd_pipeline/focal_planes/refine_centres.py [--run] [--prefix rg_]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dfdd_pipeline.common.paths import ROOT  # noqa: E402
from dfdd_pipeline.focal_planes.pattern_analysis import analyse  # noqa: E402

SWEEPS = ROOT / "artifacts/focal_planes/sweeps"
FAST = ["--n-cal", "48", "--n-test", "48", "--n-wide", "61", "--steps", "300"]
EXTRA_DELTAS = [-0.1, -0.3, -0.4]


def geometric_delta(rho: float) -> float:
    return -(np.sqrt(rho) - 1) / (np.sqrt(rho) + 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Run the extra pairs (otherwise only summarise)")
    parser.add_argument("--prefix", default="rg_")
    args = parser.parse_args()
    sweeps = sorted(p for p in SWEEPS.iterdir() if p.is_dir() and p.name.startswith(args.prefix) and (p / "results.jsonl").exists())
    if args.run:
        for sweep in sweeps:
            meta = json.loads((sweep / "meta.json").read_text(encoding="utf-8"))
            centred = [r for r in analyse(sweep) if abs(r["delta"]) < 1e-9][0]
            alphas = sorted({round(centred["alpha_star"] * f, 3) for f in (0.87, 1.0, 1.15)})
            cam = meta["camera"]
            command = [sys.executable, str(ROOT / "dfdd_pipeline/focal_planes/sweep_focal_planes.py"), "--name", sweep.name, *FAST,
                       "--lens-mm", str(cam["lens_mm"]), "--fstop", str(cam["fstop"]), "--z-lo", str(meta["z_lo"]),
                       "--z-hi", str(meta["z_hi"]), "--alphas", *map(str, alphas), "--deltas", *map(str, EXTRA_DELTAS)]
            print("+", sweep.name, alphas, flush=True)
            with (ROOT / f"artifacts/focal_planes/refine_{sweep.name}.log").open("w", encoding="utf-8") as fh:
                subprocess.run(command, check=True, stdout=fh, stderr=subprocess.STDOUT)

    lines = ["| range | rho | K*W | geometric-mean rule delta | best delta (mean) | best delta (worst) | mean at best | mean at rule-nearest |",
             "|---|---|---|---|---|---|---|---|"]
    out = []
    for sweep in sweeps:
        meta = json.loads((sweep / "meta.json").read_text(encoding="utf-8"))
        records = [json.loads(l) for l in (sweep / "results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        by_delta = defaultdict(list)
        for r in records:
            by_delta[round(r["delta"], 3)].append(r)
        # best over alpha at each delta
        best = {d: (min(rs, key=lambda r: r["test"]["0.0"]["mean_rel"]), min(rs, key=lambda r: r["test"]["0.0"]["max_rel"]))
                for d, rs in by_delta.items()}
        deltas = sorted(best)
        mean_best = min(deltas, key=lambda d: best[d][0]["test"]["0.0"]["mean_rel"])
        worst_best = min(deltas, key=lambda d: best[d][1]["test"]["0.0"]["max_rel"])
        rho = meta["z_hi"] / meta["z_lo"]
        rule = geometric_delta(rho)
        nearest = min(deltas, key=lambda d: abs(d - rule))
        row = {"sweep": sweep.name, "rho": rho, "kw_px": meta["range_blur_span_px"], "rule_delta": rule,
               "best_delta_mean": mean_best, "best_delta_worst": worst_best,
               "mean_at_best": best[mean_best][0]["test"]["0.0"]["mean_rel"],
               "mean_at_rule_nearest": best[nearest][0]["test"]["0.0"]["mean_rel"],
               "per_delta": {str(d): {"best_mean": best[d][0]["test"]["0.0"]["mean_rel"], "best_worst": best[d][1]["test"]["0.0"]["max_rel"],
                                      "alpha_at_best_mean": best[d][0]["alpha"]} for d in deltas}}
        out.append(row)
        lines.append(f"| {meta['z_lo']:g}-{meta['z_hi']:g} m | {rho:.2f} | {row['kw_px']:.2f} | {rule:+.2f} | {mean_best:+.1f} | "
                     f"{worst_best:+.1f} | {100 * row['mean_at_best']:.2f} % | {100 * row['mean_at_rule_nearest']:.2f} % ({nearest:+.1f}) |")
    (SWEEPS / "centres.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    (SWEEPS / "centres.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
