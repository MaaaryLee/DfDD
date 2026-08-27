"""Apply the small, validated SpiderCam compatibility edits idempotently."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPIDERCAM = ROOT / "vendor" / "SpiderCam"

REPLACEMENTS = {
    "datasets/__init__.py": [
        (
            "return DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=5)",
            "return DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)",
        )
    ],
    "models/focal_split.py": [
        ("nn.ParameterList([RingOpticalParameter", "nn.ModuleList([RingOpticalParameter"),
        ("nn.ParameterList([RadialOpticalParameter", "nn.ModuleList([RadialOpticalParameter"),
        ("nn.ParameterList([PixelGridOpticalParameter", "nn.ModuleList([PixelGridOpticalParameter"),
        ("nn.ParameterList([PolynomialOpticalParameter", "nn.ModuleList([PolynomialOpticalParameter"),
    ],
    "utils/metrics.py": [
        (
            "mae = (np.abs(z_true - z_pred) * mask).sum(axis=(-3,-2,-1)) / mask.sum(axis=(-3,-2,-1))\n"
            "    # mae = mae.squeeze(-1)\n"
            "    z_true_mean = z_true.mean(axis=(-3,-2,-1))",
            "mae = ((np.abs(z_true - z_pred) * mask).sum(axis=(-3,-2,-1)) / "
            "mask.sum(axis=(-3,-2,-1))).reshape(-1)\n"
            "    z_true_mean = z_true.mean(axis=(-3,-2,-1)).reshape(-1)",
        )
    ],
    "utils/visualization.py": [
        ("depth = z_true[...,0,0]", "depth = z_true[...,0,0].reshape(-1)"),
        (
            "confs = (conf * mask).sum(axis=(-3,-2,-1)) / mask.sum(axis=(-3,-2,-1))",
            "confs = ((conf * mask).sum(axis=(-3,-2,-1)) / mask.sum(axis=(-3,-2,-1))).reshape(-1)",
        ),
        (
            "mae = (np.abs(z_true - z_pred) * mask).sum(axis=(-3,-2,-1)) / mask.sum(axis=(-3,-2,-1))",
            "mae = ((np.abs(z_true - z_pred) * mask).sum(axis=(-3,-2,-1)) / "
            "mask.sum(axis=(-3,-2,-1))).reshape(-1)",
        ),
    ],
}


def main() -> None:
    for relative, replacements in REPLACEMENTS.items():
        path = SPIDERCAM / relative
        text = path.read_text(encoding="utf-8")
        for before, after in replacements:
            if after in text:
                continue
            count = text.count(before)
            if count == 0:
                raise RuntimeError(f"Expected source text not found in {relative}: {before[:70]}")
            text = text.replace(before, after)
        path.write_text(text, encoding="utf-8")
        print(f"Prepared {relative}")


if __name__ == "__main__":
    main()
