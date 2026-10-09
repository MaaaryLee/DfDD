"""Fast simulator of the DfDD slide calibration for arbitrary optics and focal-plane pairs.

Physics. For a thin lens of focal length F and f-number N, a point at distance z imaged with focus
at distance s has a blur circle of diameter ``K * |1/s - 1/z|`` pixels, with

    K = F^2 / (N * pixel_pitch)        (pixels per dioptre).

The calibration slide is rescaled at every depth to fill the view, so its sharp image is the same at
every depth, and a slide image is fully described by its blur diameter. ``SlideRenderer`` therefore
renders any slide image from one supersampled Blender render of the sharp slide
(``render_reference_slide.py``): it convolves the scene-linear image with the aperture disk, applies
Cycles' Blackman-Harris pixel filter, samples the nominal pixel grid, maps the result through the
view transform fitted pixel by pixel to Blender's own PNG of the same render, and adds render noise.

Estimator. ``calibrate`` trains SpiderCam's unchanged ``FocalSplit`` model (2 scales, dx/dy, joint,
ring constants, VW confidence) with the trainer's optimiser and loss. ``FastFilteredDepthLoss`` is
``FilteredDepthLoss`` with the per-ring quantiles taken over each ring's pixels only (identical
values, ~16x less sorting). ``physics_init`` starts B at the dioptric midpoint of the focal planes
and A at the least-squares value of ``Is = -A * L * (1/z - B)``, which the trained constants follow
(see ``docs/FOCAL_PLANES.md``), so a calibration converges in a few hundred steps instead of 2000.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import Tensor, nn

from dfdd_pipeline.common.paths import SPIDERCAM  # noqa: F401  (puts SpiderCam on sys.path)
from models import FocalSplit  # noqa: E402
from utils.torch import get_pixel_coords  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOADER_GRAY = (0.2989, 0.587, 0.114)  # rgb_to_grayscale applied to cv2's BGR tensor: weights on (B, G, R)
CROP = 11  # the trainer and the analysis ignore an 11-pixel border


# --------------------------------------------------------------------------------------- optics
@dataclass(frozen=True)
class Camera:
    lens_mm: float = 85.0
    fstop: float = 2.0
    sensor_width_mm: float = 36.0
    width: int = 320
    height: int = 240

    @property
    def pitch_mm(self) -> float:
        return self.sensor_width_mm / self.width

    @property
    def k_px_per_diopter(self) -> float:
        """Blur-circle diameter (pixels) per dioptre of defocus."""
        return self.lens_mm ** 2 / (self.fstop * self.pitch_mm) * 1e-3

    def blur_px(self, focus_m, depth_m):
        return self.k_px_per_diopter * np.abs(1.0 / np.asarray(focus_m) - 1.0 / np.asarray(depth_m))


def diopter_design(z_lo: float, z_hi: float, alpha: float, delta: float = 0.0) -> tuple[float, float]:
    """Focal planes for target range [z_lo, z_hi] (m): centre c0 + delta*W, half-separation alpha*W.

    c0 and W are the dioptric centre and half-width of the range; alpha = 1, delta = 0 puts the
    focal planes exactly at the range ends. Returns (near_focus_m, far_focus_m); a far plane at or
    beyond infinity is returned as inf.
    """
    u_lo, u_hi = 1.0 / z_hi, 1.0 / z_lo
    c0, w = (u_lo + u_hi) / 2.0, (u_hi - u_lo) / 2.0
    c, h = c0 + delta * w, alpha * w
    near, far = c + h, c - h
    return 1.0 / near, (1.0 / far if far > 0 else math.inf)


# ------------------------------------------------------------------------------------- renderer
def _blackman_harris_otf(freq: Tensor, width: float) -> Tensor:
    """Fourier transform (unit DC) of the centred Blackman-Harris window of the given width."""
    a = (0.35875, 0.48829, 0.14128, 0.01168)
    wf = width * freq
    total = a[0] * torch.sinc(wf)
    for m in (1, 2, 3):
        total = total + 0.5 * a[m] * (torch.sinc(wf - m) + torch.sinc(wf + m))
    return total / a[0]


class SlideRenderer:
    """Slide images at arbitrary blur diameters from ``render_reference_slide.py`` output."""

    def __init__(self, reference_dir: Path, tone: str = "agx", psf: str = "disk", device: torch.device = DEVICE):
        reference_dir = Path(reference_dir)
        meta = json.loads((reference_dir / "metadata.json").read_text(encoding="utf-8"))
        self.meta, self.tone, self.psf, self.device = meta, tone, psf, device
        if psf not in ("disk", "gaussian"):
            raise ValueError(f"Unknown psf {psf!r}")
        self.ss = int(meta["sharp"]["supersample"])
        self.wide_w, self.wide_h = meta["sharp"]["wide_size"]
        self.width, self.height = meta["resolution"]
        self.off_y, self.off_x = (self.wide_h - self.height) // 2, (self.wide_w - self.width) // 2

        linear = np.load(reference_dir / "sharp_linear.npy").astype(np.float32)  # (H, W, 3) RGB
        png = cv2.imread(str(reference_dir / "sharp.png"), cv2.IMREAD_UNCHANGED)[..., ::-1].astype(np.float32) / 257.0
        self.lut_x, self.lut_y = self._fit_tone_curve(linear, png)
        if tone == "linear":
            # Linear sensor: gain chosen so the brightest 0.1 % of the sharp slide sits at 230 grey levels.
            gray = linear[..., 2] * LOADER_GRAY[0] + linear[..., 1] * LOADER_GRAY[1] + linear[..., 0] * LOADER_GRAY[2]
            self.linear_gain = 230.0 / float(np.quantile(gray, 0.999))
        elif tone != "agx":
            raise ValueError(f"Unknown tone {tone!r}")

        sharp = torch.from_numpy(linear).permute(2, 0, 1).contiguous().to(device)
        self.hires_shape = sharp.shape[-2:]
        self.spectrum = torch.fft.rfft2(sharp)
        fy = torch.fft.fftfreq(self.hires_shape[0], device=device)
        fx = torch.fft.rfftfreq(self.hires_shape[1], device=device)
        self.rho = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
        # Cycles' default pixel filter: Blackman-Harris, filter_width 1.5, which Cycles doubles
        # internally (film.cpp), so the window spans 3 nominal pixels; separable in x and y.
        # validate_simulator.py: RMS error at zero blur 0.99 grey with 1.5 px, 0.20 with 3 px.
        width = 3.0 * self.ss
        pixel_filter = _blackman_harris_otf(fy, width)[:, None] * _blackman_harris_otf(fx, width)[None, :]
        shift = (self.ss - 1) / 2.0  # sample at nominal pixel centres, half-way between hi-res pixels
        phase = torch.exp(2j * math.pi * (fy[:, None] + fx[None, :]) * shift)
        self.base_otf = pixel_filter * phase

    @staticmethod
    def _fit_tone_curve(linear: np.ndarray, png: np.ndarray, bins: int = 400):
        """Per-channel monotone map from scene-linear value to PNG grey level (0-255)."""
        lut_x, lut_y = [], []
        for c in range(3):
            x, y = linear[..., c].ravel(), png[..., c].ravel()
            edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, bins + 1)))
            index = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
            count = np.bincount(index, minlength=len(edges) - 1)
            keep = count > 0
            mean_x = np.bincount(index, x, len(edges) - 1)[keep] / count[keep]
            mean_y = np.bincount(index, y, len(edges) - 1)[keep] / count[keep]
            mean_y = np.maximum.accumulate(mean_y)
            lut_x.append(mean_x)
            lut_y.append(mean_y)
        return lut_x, lut_y

    def linear_images(self, blur_px: Tensor, chunk: int = 8) -> Tensor:
        """(N, 3, H, W) scene-linear nominal-resolution images for blur diameters ``blur_px`` (pixels)."""
        blur_px = torch.as_tensor(blur_px, dtype=torch.float32, device=self.device).reshape(-1)
        out = []
        for start in range(0, blur_px.numel(), chunk):
            d = blur_px[start:start + chunk, None, None] * self.ss
            if self.psf == "disk":  # uniform circular aperture (Cycles, no blades)
                arg = math.pi * d * self.rho[None]
                otf = torch.where(arg > 1e-6, 2.0 * torch.special.bessel_j1(arg) / arg.clamp_min(1e-6), torch.ones_like(arg))
            else:  # Gaussian with the disk's second moment (sigma = diameter / 4): the heat-equation ideal
                otf = torch.exp(-2.0 * math.pi ** 2 * (d / 4.0) ** 2 * self.rho[None] ** 2)
            spec = self.spectrum[None] * (otf * self.base_otf[None])[:, None]
            image = torch.fft.irfft2(spec, s=self.hires_shape)[..., :: self.ss, :: self.ss]
            out.append(image[..., self.off_y:self.off_y + self.height, self.off_x:self.off_x + self.width])
        return torch.cat(out)

    def _tone(self, linear: Tensor) -> Tensor:
        channels = []
        for c in range(3):
            x = torch.as_tensor(self.lut_x[c], dtype=torch.float32, device=self.device)
            y = torch.as_tensor(self.lut_y[c], dtype=torch.float32, device=self.device)
            v = linear[:, c].contiguous()
            idx = torch.clamp(torch.searchsorted(x, v) - 1, 0, x.numel() - 2)
            x0, x1, y0, y1 = x[idx], x[idx + 1], y[idx], y[idx + 1]
            t = ((v - x0) / (x1 - x0).clamp_min(1e-12)).clamp(0.0, 1.0)
            channels.append(y0 + t * (y1 - y0))
        return torch.stack(channels, dim=1)

    def gray_levels(self, blur_px: Tensor) -> Tensor:
        """(N, 1, H, W) noise-free grey levels (0-255) as SpiderCam's loader forms them."""
        linear = self.linear_images(blur_px)
        if self.tone == "linear":
            rgb = linear * self.linear_gain
        else:
            rgb = self._tone(linear)
        # cv2 loads BGR and rgb_to_grayscale weights channel 0 as red: weights act on (B, G, R).
        gray = LOADER_GRAY[0] * rgb[:, 2] + LOADER_GRAY[1] * rgb[:, 1] + LOADER_GRAY[2] * rgb[:, 0]
        return gray[:, None]


@dataclass
class NoiseModel:
    sigma: float = 1.0          # grey levels (0-255) per image, white
    pair_correlation: float = 0.0  # correlation of the noise of the two focus images (shared seed)
    quantize: str = "16bit"     # "16bit" (full-precision loader) or "8bit" (upstream loader)


def to_model_input(gray: Tensor, quantize: str) -> Tensor:
    """Grey levels -> the loader's normalised value (PNG/257 with the full-precision wrapper, /256)."""
    if quantize == "8bit":
        return torch.round(gray.clamp(0, 255) * 257.0 / 256.0).clamp(0, 255) / 256.0
    if quantize == "16bit":
        return torch.round(gray.clamp(0, 255) * 257.0) / 257.0 / 256.0
    if quantize == "none":
        return gray / 256.0
    raise ValueError(quantize)


def slide_sweep(renderer: SlideRenderer, camera: Camera, focus_near: float, focus_far: float, depths,
                noise: NoiseModel, generator: torch.Generator | None = None):
    """(I_plus, I_minus, z) as LinearSlideDatasetNew returns them: I_plus = far-focus image (cam_1)."""
    depths = np.asarray(depths, dtype=np.float64)
    b_far = camera.blur_px(focus_far, depths)
    b_near = camera.blur_px(focus_near, depths)
    mean_far = renderer.gray_levels(torch.as_tensor(b_far, dtype=torch.float32))
    mean_near = renderer.gray_levels(torch.as_tensor(b_near, dtype=torch.float32))
    shape = mean_far.shape
    gen = dict(generator=generator, device=mean_far.device)
    rho = noise.pair_correlation
    shared = torch.randn(shape, **gen) if rho > 0 else 0.0
    n_far = noise.sigma * (math.sqrt(rho) * shared + math.sqrt(1 - rho) * torch.randn(shape, **gen))
    n_near = noise.sigma * (math.sqrt(rho) * shared + math.sqrt(1 - rho) * torch.randn(shape, **gen))
    i_plus = to_model_input(mean_far + n_far, noise.quantize)
    i_minus = to_model_input(mean_near + n_near, noise.quantize)
    z = torch.as_tensor(depths, dtype=torch.float32, device=mean_far.device).reshape(-1, 1)
    return i_plus, i_minus, z


# --------------------------------------------------------------------------------- calibration
def focal_split_config(height: int = 240, width: int = 320, n_rings: int = 4, sparsity: float = 0.9,
                       lr: float = 0.05, weight_decay: float = 0.005, box_denoise: int = 0, n_scales: int = 2):
    """The configuration ``run_checkerboard_calibration.py`` trains (2 scales, dx/dy, joint, rings, VW)."""
    return OmegaConf.create({
        "data": {"mask": [0, height, 0, width], "center": [height // 2, width // 2]},
        "model": {"name": "focal_split", "n_scales": n_scales, "dxdy": True, "const": "rings", "n_rings": n_rings,
                  "n_orders": 2, "conf": "VW", "gaussian_denoise": 5, "box_filter_background": 3,
                  "box_denoise": box_denoise or False, "mode": "joint", "fft": False},
        "optim": {"lr": lr, "weight_decay": weight_decay, "clip": 1.0, "sparsity": sparsity,
                  "z_min": [0.0, 0.0], "z_max": [1.5, 1.5]},
    })


class FastFilteredDepthLoss(nn.Module):
    """``models.loss.FilteredDepthLoss`` (L1 on the top (1-sparsity) confidence of every ring)."""

    def __init__(self, config):
        super().__init__()
        self.sparsity = float(config.optim.sparsity)
        shape = (config.data.mask[1] - config.data.mask[0] - 2 * CROP, config.data.mask[3] - config.data.mask[2] - 2 * CROP)
        X, Y = get_pixel_coords(shape, config.data.center)
        R = torch.sqrt(X ** 2 + Y ** 2)
        r_list = np.linspace(0.0, R.max(), config.model.n_rings + 1)
        flat = R.reshape(-1)
        self.ring_index = [torch.nonzero((flat >= r_list[i]) & (flat < r_list[i + 1])).reshape(-1)
                           for i in range(config.model.n_rings)]
        for i, index in enumerate(self.ring_index):
            self.register_buffer(f"ring_{i}", index, persistent=False)

    def selection(self, conf: Tensor) -> Tensor:
        flat = conf.reshape(conf.shape[0], -1)
        mask = torch.zeros_like(flat, dtype=torch.bool)
        for i in range(len(self.ring_index)):
            index = getattr(self, f"ring_{i}")
            values = flat[:, index]
            bound = torch.quantile(values, self.sparsity, dim=1, keepdim=True)
            mask[:, index] = values >= bound
        return mask.reshape(conf.shape)

    def forward(self, z_pred: Tensor, z_true: Tensor, conf: Tensor) -> Tensor:
        mask = self.selection(conf.detach())
        return torch.abs(z_pred - z_true)[mask].mean()


def _preprocess(model: FocalSplit, image: Tensor) -> Tensor:
    if model.model_config.box_filter_background:
        image = image - F.conv2d(image, model.box_filter_background, padding="same")
    if model.model_config.gaussian_denoise:
        image = model.gaussian_denoise(image)
    return image


@torch.no_grad()
def physics_init(model: FocalSplit, i_plus: Tensor, i_minus: Tensor, z: Tensor, b_value: float) -> None:
    """Set every ring of B to the dioptric midpoint and A to the least-squares Is = -A L (1/z - B)."""
    p_levels, p_laps = model.pyramid(_preprocess(model, i_plus))
    m_levels, m_laps = model.pyramid(_preprocess(model, i_minus))
    u = (1.0 / z).reshape(-1, 1, 1, 1)
    for scale, (A, B) in enumerate(zip(model.A_list, model.B_list)):
        lap = (p_laps[scale] + m_laps[scale]) / 2
        i_s = (p_levels[scale] - m_levels[scale]) / 2
        c = CROP // 2 ** scale + 1
        lap, i_s = lap[..., c:-c, c:-c], i_s[..., c:-c, c:-c]
        x = lap * (u - b_value)
        a_value = float(-(i_s * x).sum() / (x * x).sum().clamp_min(1e-20))
        A.x.data.fill_(a_value)
        B.x.data.fill_(b_value)


@dataclass
class CalibrationResult:
    model: FocalSplit
    losses: list[float]
    seconds: float


def calibrate(i_plus: Tensor, i_minus: Tensor, z: Tensor, config, steps: int = 400, init: str = "physics",
              b_value: float | None = None, log_every: int = 0, schedule: str = "constant") -> CalibrationResult:
    """Train FocalSplit full-batch as ``train_focal_split.py`` does (one AdamW step per epoch).

    ``schedule="constant"`` is the trainer's fixed learning rate; it never settles (the loss keeps
    bouncing), so repeated calibrations of the same images differ by a few tenths of a percent.
    ``schedule="cosine"`` decays the rate to 1 % of its start so every calibration converges.
    """
    import time

    torch.manual_seed(0)
    model = FocalSplit(config).to(i_plus.device)
    if init == "physics":
        if b_value is None:
            raise ValueError("physics init needs the dioptric midpoint b_value")
        physics_init(model, i_plus, i_minus, z, b_value)
    loss_fn = FastFilteredDepthLoss(config).to(i_plus.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.optim.lr, weight_decay=config.optim.weight_decay)
    scheduler = None
    if schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=0.01 * config.optim.lr)
    elif schedule != "constant":
        raise ValueError(schedule)
    z_map = z.reshape(-1, 1, 1, 1)
    losses = []
    start = time.time()
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        z_pred, conf, *_ = model(i_plus, i_minus)
        zp = z_pred[..., CROP:-CROP, CROP:-CROP]
        loss = loss_fn(zp, z_map.expand_as(zp), conf[..., CROP:-CROP, CROP:-CROP])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.optim.clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(float(loss.detach()))
        if log_every and step % log_every == 0:
            print(f"    step {step}: loss {losses[-1]:.4f}")
    if i_plus.is_cuda:
        torch.cuda.synchronize()
    return CalibrationResult(model, losses, time.time() - start)


@torch.no_grad()
def predict(model: FocalSplit, i_plus: Tensor, i_minus: Tensor, chunk: int = 24):
    """Cropped (z_pred, conf) arrays (N, 1, h, w) as the analysis reads them."""
    preds, confs = [], []
    model.eval()
    for start in range(0, i_plus.shape[0], chunk):
        z_pred, conf, *_ = model(i_plus[start:start + chunk], i_minus[start:start + chunk])
        preds.append(z_pred[..., CROP:-CROP, CROP:-CROP].float().cpu())
        confs.append(conf[..., CROP:-CROP, CROP:-CROP].float().cpu())
    model.train()
    return torch.cat(preds).numpy(), torch.cat(confs).numpy()


def per_depth_errors(z_pred: np.ndarray, conf: np.ndarray, depths, sparsities=(0.0, 0.9)) -> dict:
    """MAE (m) of each frame on the pixels kept at each sparsity (analysis.confidence_mask rule)."""
    depths = np.asarray(depths, dtype=np.float64)
    out = {}
    flat = conf.reshape(conf.shape[0], -1)
    err = np.abs(z_pred.reshape(z_pred.shape[0], -1) - depths[:, None])
    for s in sparsities:
        if s == 0.0:
            mae = err.mean(axis=1)
            med = np.median(z_pred.reshape(z_pred.shape[0], -1), axis=1)
        else:
            bound = np.quantile(flat, s, axis=1, keepdims=True)
            keep = flat >= bound
            mae = (err * keep).sum(axis=1) / np.maximum(keep.sum(axis=1), 1)
            med = np.array([np.median(z_pred.reshape(z_pred.shape[0], -1)[i][keep[i]]) for i in range(len(depths))])
        out[s] = {"mae_m": mae, "median_pred_m": med}
    return out


def longest_interval(depths, ok) -> tuple[float, float] | None:
    best, start = None, None
    for i, good in enumerate(list(ok) + [False]):
        if good and start is None:
            start = i
        elif not good and start is not None:
            if best is None or (i - start) > (best[1] - best[0] + 1):
                best = (start, i - 1)
            start = None
    return None if best is None else (float(depths[best[0]]), float(depths[best[1]]))


def summarize_range(depths, mae, z_lo: float, z_hi: float, threshold: float = 0.10) -> dict:
    """Range metrics: relative MAE over [z_lo, z_hi], worst depth, 10 % coverage, working interval."""
    depths, mae = np.asarray(depths), np.asarray(mae)
    rel = mae / depths
    inside = (depths >= z_lo - 1e-9) & (depths <= z_hi + 1e-9)
    interval = longest_interval(depths, rel < threshold)
    return {
        "mean_rel": float(rel[inside].mean()),
        "max_rel": float(rel[inside].max()),
        "mae_m": float(mae[inside].mean()),
        "coverage": float((rel[inside] < threshold).mean()),
        "working_range_m": interval,
    }
