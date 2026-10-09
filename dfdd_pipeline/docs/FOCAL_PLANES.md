# Where to put the two focal planes for a given working range

**Question.** To make DfDD work over a target range, for example 2.0-4.3 m (about the depth of a
room), where should the two focus distances go? At the range ends, inside them, or outside?

**Answer in brief.**

* **For the room camera (85 mm f/2) and 2.0-4.3 m, the planes belong inside the range**, at about
  2.4-2.5 m and 3.9-4.1 m. On the calibration slide this halves the error of the pair used so far
  (1.984 / 4.416 m): mean 2.15 % instead of 4.37 %, worst depth 4.7 % instead of 9.8 %, and the
  10 % working range grows from 1.85-4.28 m to 1.73-5.06 m. Blender renders confirm the simulator
  pair by pair to 0.01 points. A pair at 2.3 / 4.0 m is within 0.2 points of the best.
* **It carries over to the rooms.** On the 16 room views rendered again at 2.3 / 4.0 m, DfDD on
  denoised pairs improves by 22 % (30.8 -> 24.1 cm in the 2.08-4.32 m band, better in 12 of 15
  views) and the confident pixels of raw pairs by 9 %; only the dense map of raw, noisy pairs is
  unchanged, because there the render noise dominates (section 3.5).
* **The general rule is set by blur in pixels, not by distance.** Work in dioptres. Choose the
  separation so that an object in focus on one sensor is a blur disk of about 5 px on the other
  (half-separation K*h = 2.5 px, K = F^2 / (N * pixel pitch)), and centre the pair on the far side
  of the range's dioptric centre, between the geometric mean and the metric midpoint of the range. This held for six apertures (f/1.4-f/8) and six ranges
  (1-1.6 m to 4-12 m): the best half-separation was always 2.1-2.8 px, while the best separation
  relative to the range varied six-fold (0.32-1.95). Large apertures therefore want the planes inside the
  range, small apertures outside it, and "planes on the range ends" is right only when the range
  spans about 3 px of blur.
* **Noise moves and flattens the optimum.** With 1-2 grey levels of sensor noise the best
  half-separation grows to 3.2-3.7 px and the choice of planes matters little; the noise does.
* `focal_planes/design_focal_planes.py` recommends a pair for any camera and range in about five
  minutes, using a slide simulator validated against Blender.

## 1. Framework: think in dioptres and in pixels of blur

For a thin lens with focal length F, f-number N and pixel pitch p, a point at distance z imaged with
focus at distance s has a blur-circle diameter

    b = K * |1/s - 1/z|  pixels,   K = F^2 / (N p)   (pixels per dioptre, F and p in metres).

| camera (320 px across 36 mm) | K (px/D) |
|---|---|
| 35 mm f/1.4 | 7.8 |
| 35 mm f/0.7 | 15.6 |
| 85 mm f/2 | 32.1 |

Everything about the focus pair is therefore linear in inverse depth u = 1/z. Write a target range
[z_lo, z_hi] as its dioptric centre c0 = (1/z_lo + 1/z_hi)/2 and half-width W = (1/z_lo - 1/z_hi)/2,
and a focus pair as its dioptric centre c and half-separation h. Two dimensionless numbers then
describe any design:

* `alpha = h / W`: separation relative to the range (alpha = 1 puts the planes on the range ends,
  alpha < 1 moves both inward);
* `delta = (c - c0) / W`: offset of the pair's centre from the range's dioptric centre (delta < 0 is
  toward the far end).

The 2.0-4.3 m range has c0 = 0.366 D (2.73 m, not the metric middle 3.15 m) and W = 0.134 D. Planes
pulled inside the range to 2.3 / 4.0 m are alpha = 0.69, delta = -0.18; the room-scale runs so far
used 1.984 / 4.416 m, alpha = 1.04, delta = -0.01.

**The calibrated constants follow this picture.** FocalSplit predicts `1/z = B - Is / (A L)`. In
the heat-equation limit `Is = (K^2 h / 16) (1/z - c) * Laplacian(I)`, so B should equal the dioptric
midpoint c of the two focal planes and A should grow like K^2 h. Every existing checkpoint agrees:

| checkpoint | focus (m) | dioptric midpoint | calibrated B (scale 1) |
|---|---|---|---|
| 85 mm f/2 | 1.984 / 4.416 | 0.365 | 0.370 |
| 35 mm f/0.7 native | 1.984 / 4.416 | 0.365 | 0.374 |
| 35 mm f/1.4 native | 1.984 / 4.416 | 0.365 | 0.364 |
| 35 mm f/1.4 | 0.62 / 1.38 | 1.169 | 1.172 |
| 35 mm f/1.4 | 1.1 / 6.5 | 0.532 | 0.51 |

and A at f/0.7 is 3.4x (scale 1) to 4.1x (scale 2) its value at f/1.4, against 4x predicted by K^2.

## 2. Method: a fast, validated slide simulator

The calibration slide is rescaled at every depth to fill the view, so its sharp image is the same
at every depth and a slide image depends only on its blur diameter. `focal_planes/simulator.py`
renders any slide image from one supersampled Blender render of the sharp slide
(`focal_planes/render_reference_slide.py`): aperture-disk convolution of the scene-linear image,
Cycles' pixel filter, the AgX view transform fitted pixel by pixel to Blender's own PNG, render
noise. The estimator is SpiderCam's unchanged `FocalSplit` (2 scales, dx/dy, joint, 4 rings, VW
confidence), trained with the trainer's optimiser and loss.

Validation (`focal_planes/validate_simulator.py`, `focal_planes/check_calibration.py`):

* **Images.** Against Blender renders at known blur diameters the simulator is within 0.20 grey
  levels at zero blur and at the render-noise floor (0.05) from 4 px on; against all 142 images of
  the existing 85 mm f/2 slide sweep, median 0.08 and worst 0.22 grey levels (slide texture contrast:
  9 grey levels standard deviation). This needed one correction: Cycles doubles the Blackman-Harris
  filter width internally, so the pixel filter spans 3 px, not 1.5 (0.99 -> 0.20 grey at zero blur).
* **Render noise of the slides is tiny**: 0.02-0.07 grey levels at 2048 samples, and 30-50 %
  correlated between renders that share Cycles' default seed. The existing slide calibrations are
  therefore effectively noise-free; real sensors are not (section 3.4).
* **Calibrations.** Same slide sweep, calibrated on Blender images vs simulated images:

| sweep | Blender: dense / 90 % MAE | simulator: dense / 90 % MAE |
|---|---|---|
| 85 mm f/2, focus 1.984 / 4.416, 2.08-4.32 m | 15.4 / 11.7 cm | 15.3 / 11.6 cm |
| 35 mm f/0.7, same focus and range | 11.5 / 8.6 cm | 11.5 / 8.7 cm |
| 35 mm f/1.4, same focus and range | 14.3 / 10.7 cm | 13.8 / 10.8 cm |
| 35 mm f/1.4, focus 0.62 / 1.38, 0.65-1.35 m | 3.2 / 2.5 cm | 3.2 / 2.5 cm |

* **Faster calibration.** `FastFilteredDepthLoss` returns exactly the value of SpiderCam's
  `FilteredDepthLoss`, about 16x faster. With SpiderCam's own initialisation and 2000 steps the fast
  trainer reproduces the pipeline's 85 mm checkpoint (20.1 cm dense, as in its `RUN_SUMMARY.md`).
  Starting B at the dioptric midpoint and A at its least-squares value (`physics_init`) reaches a
  better optimum within 200 steps: 15.4 cm dense, 11.7 cm at 90 %. The 85 mm checkpoint in the repo
  is under-converged by about 25 %.

A simulated calibration takes about 10 s instead of about 35 minutes of rendering and training,
so a focal-plane search is a few minutes of GPU time.

## 3. Results

### 3.1 The room camera: 85 mm f/2, target 2.0-4.3 m

**The best pair is inside the range: about 2.46 / 4.07 m, half the error of planes on the range
ends.** Mean relative error over the range 2.15 % (worst depth 4.7 %) against 4.27 % (9.8 %) with
the planes on 2.0 / 4.3 m and 4.37 % (9.8 %) with the 1.984 / 4.416 m pair used so far. Planes at
2.3 / 4.0 m are within 0.2 points of the best: 2.37 % (4.8 %).

Two regularities (`artifacts/focal_planes/sweeps/f2_2to4p3_main/alpha_curves.png`, 75 pairs):

* **The best separation does not depend on where the pair is centred:** alpha = 0.60 at every
  centre offset from 0 to -0.4 W, i.e. the planes sit at 60 % of the range's dioptric half-width
  from its centre. In pixels this is a half-separation blur K*h of 2.6 px. Too close (alpha 0.3)
  costs about 1.3x the best error, planes on the range ends (alpha 1.0) about 1.6x.
  Scoring only the top-10 % confidence pixels gives the same optimum (alpha 0.55-0.6, centre -0.3 to
  -0.4 W: 1.6 % mean, 3.5 % worst, against 3.3 % / 8.1 % for 1.984 / 4.416 m).
* **Centre the pair on the far side of the range's dioptric centre:** moving the centre from 0 to
  -0.3 W lowers the mean error from 2.70 % to 2.15 %; the worst-depth error is lowest at -0.2 to
  -0.3 W (4.7-4.8 %) and rises again at -0.4 W. The far end carries most of the error because a
  given dioptric error is z times larger in relative depth error there.

| pair (m) | alpha | centre | mean error | worst depth | 90 % sparse mean |
|---|---|---|---|---|---|
| 1.984 / 4.416 (used so far) | 1.04 | -0.01 W | 4.37 % | 9.8 % | 3.29 % |
| 2.00 / 4.30 (range ends) | 1.00 | 0 | 4.27 % | 9.8 % | 3.25 % |
| 2.24 / 3.50 | 0.60 | 0 | 2.70 % | 5.9 % | 2.13 % |
| 2.30 / 4.00 | 0.69 | -0.18 W | 2.37 % | 4.8 % | 1.90 % |
| **2.46 / 4.07** | **0.60** | **-0.30 W** | **2.15 %** | **4.7 %** | **1.67 %** |
| 2.46 / 3.07 (too close) | 0.30 | 0 | 3.48 % | 9.9 % | 2.54 % |

**The working range ends beyond the focal planes.** Scored on a wider sweep than it was calibrated
on, the 2.46 / 4.07 m pair stays under the 10 % criterion from 1.73 m to 5.06 m, while the
range-end pair 2.0 / 4.3 m reaches 10 % at 1.85 m and 4.28 m (`error_vs_depth.png`). In blur
pixels, the 10 % working range is widest, about +-6 px around its centre, for half-separations of
2.4-3.2 px; it shrinks to +-4.3-4.9 px for wider pairs. So the working range is always wider than
the focal-plane separation (about 2.4x the half-separation at the optimum), and it reaches further
past the near plane than past the far one because 10 % of depth is a larger dioptric tolerance up
close. Beyond the far end every pair compresses depth toward a ceiling (`response_curves.png`);
the wider the separation, the lower the ceiling.

**Blender confirms it** (`artifacts/focal_planes/blender/f2_2p08to4p32/blender_vs_sim.png`): four
pairs rendered as real Blender slide sweeps (2.08-4.32 m) and calibrated the same way:

| pair (m) | Blender: mean / worst | simulator: mean / worst |
|---|---|---|
| 1.984 / 4.416 | 4.34 % / 9.5 % | 4.34 % / 9.6 % |
| 2.24 / 3.50 | 2.70 % / 5.7 % | 2.70 % / 5.7 % |
| 2.30 / 4.00 | 2.27 % / 4.7 % | 2.28 % / 4.7 % |
| 2.46 / 3.07 | 3.41 % / 9.7 % | 3.41 % / 9.7 % |

**Calibration note.** With the trainer's constant learning rate (0.05) the loss never settles:
recalibrating the same images with 150-1000 steps gives 2.34-2.65 %, and a separate run 2.88 %.
That run-to-run scatter (about +-0.3 points) is as large as the differences between neighbouring
focal pairs, so the sweeps use a cosine-decayed rate (`calibrate(..., schedule="cosine")`), after
which repeated calibrations agree to 0.01-0.02 points. The same change would make the pipeline's
own checkpoints reproducible.

### 3.2 The pattern: the best separation is a fixed blur in pixels, not a fixed fraction of the range

Same target range (2.0-4.3 m), 85 mm lens at f/1.4 to f/8, so the range's blur span K*W runs from
6.1 px down to 1.1 px (`artifacts/focal_planes/sweeps/collapse_ap.png`, `pattern.png`):

| aperture | K*W (px) | best alpha (centred) | best half-separation K*h (px) | best mean error | planes on range ends |
|---|---|---|---|---|---|
| f/1.4 | 6.13 | 0.38 | 2.4 | 2.96 % | 5.29 % |
| f/2 | 4.29 | 0.58 | 2.5 | 2.71 % | 4.30 % |
| f/2.8 | 3.07 | 0.95 | 2.9 | 2.85 % | 2.86 % |
| f/4 | 2.15 | 1.26 | 2.7 | 3.18 % | 3.22 % |
| f/5.6 | 1.53 | 1.39 | 2.1 | 3.37 % | 3.44 % |
| f/8 | 1.07 | 1.85 | 2.0 | 3.50 % | 3.85 % |

(centre offset -0.2 W gives the same best alpha within 0.1 and errors 0.4-0.6 points lower.)

The same holds when the range changes instead of the aperture (85 mm f/2; `collapse_rg.png`):

| range (m) | far/near | K*W (px) | best alpha (-0.2 W) | best K*h (px) | best mean error | planes on range ends |
|---|---|---|---|---|---|---|
| 2.5-3.5 | 1.40 | 1.83 | 1.39 | 2.5 | 1.01 % | 1.05 % |
| 3-6 | 2.00 | 2.68 | 1.01 | 2.7 | 2.19 % | 2.20 % |
| 4-12 | 3.00 | 2.68 | 1.01 | 2.7 | 4.70 % | 4.70 % |
| 1.2-2.0 | 1.67 | 5.35 | 0.46 | 2.4 | 1.40 % | 2.94 % |
| 1.0-1.6 | 1.60 | 6.02 | 0.40 | 2.4 | 1.35 % | 2.75 % |
| 1.5-6 | 4.00 | 8.03 | 0.32 | 2.6 | 7.55 % | 11.31 % |

Across all twelve scenarios (six apertures, six ranges, K*W from 1.1 to 8 px) the best
half-separation at the recommended centre is 2.1-2.8 px of blur (`pattern.png`), while the best
alpha varies from 0.32 to 1.95. 3-6 m and 4-12 m have the same K*W but different far/near ratios
and the same best alpha: the separation is set by K*W, and the ratio only changes how much error is
left. Very wide ranges (1.5-6 m, 4:1) still follow the rule but cannot be served well by one pair
(7.5 % at best).

**This is what the scaling argument predicts.** A slide image depends on the optics and the depth
only through its blur diameter in pixels, so for a target range the relative error of any design
can depend only on alpha, delta, the range's blur span K*W, its far/near ratio and the noise level.
The sweeps add one empirical fact: the optimum in pixels hardly moves.

Plotted against alpha the six error curves have their minima anywhere from 0.38 to 1.85; plotted
against the half-separation in pixels of blur, K*h, all six minima sit at 2-3 px. So:

* the planes belong **inside** the range when the aperture is large (K*W above about 3 px: our
  85 mm f/2 room camera, alpha = 0.6), **on the range ends** when K*W is about 3 px (f/2.8 here),
  and **outside** the range when the aperture is small (f/4 and smaller);
* with the separation set this way, f/1.4 to f/2.8 reach the same accuracy (2.3-2.4 % at centre
  -0.2 W); a larger aperture only helps if the planes are moved closer together to match it. With
  the planes on the range ends, f/1.4 is almost twice as bad as f/2.8.

### 3.3 Why a fixed blur

Switching off one effect at a time (85 mm f/2, 2.0-4.3 m, centred pairs; `ablations.png`):

| variant | alpha 0.2 | alpha 0.6 | alpha 1.0 | alpha 2.5 | best alpha |
|---|---|---|---|---|---|
| as rendered (AgX, render noise, disk blur) | 3.90 % | 2.71 % | 4.30 % | 8.44 % | 0.6 |
| linear sensor | 3.78 % | 2.63 % | 4.41 % | 7.83 % | 0.6 |
| linear, no noise, no quantisation | 3.77 % | 2.63 % | 4.40 % | 7.83 % | 0.6 |
| Gaussian blur instead of the disk | 3.08 % | 2.68 % | 3.31 % | 3.19 % | 0.5-0.6 |
| Gaussian, linear, no noise | 2.99 % | 2.58 % | 3.38 % | 2.77 % | 0.5 |
| three pyramid scales | 4.11 % | 2.95 % | 4.22 % | 8.31 % | 0.6 |
| f/4 on 1.0-2.15 m (same K*W, same far/near ratio) | 3.90 % | 2.71 % | 4.30 % | 8.44 % | 0.6 |

* **Wide pairs fail because real defocus is a disk.** An object near one focal plane is a disk of
  diameter 2Kh in the other image. A disk blur, unlike a Gaussian, drops to zero contrast and
  reverses at some spatial frequencies (the first zero is at 1.22/(2Kh) cycles/px, inside the
  estimator's finest band once 2Kh is about 5 px), which the heat-equation model behind DfDD
  cannot represent. With a Gaussian blur of the same size the penalty for wide pairs almost
  disappears (alpha 2.5: 3.2 % instead of 8.4 %).
* **Narrow pairs fail at the range ends, and not because of noise or the tone curve.** The rise
  for small alpha survives a linear sensor with no noise and no quantisation, and with Gaussian blur.
  It sits at the range ends, mostly the far end (alpha 0.2, linear, noise-free: bias -7.7 % and
  11.7 % error at 4.3 m against 0.8 % at the centre). There both images are blurred by about the
  range's blur span and differ only by the small separation; the defocus difference the estimator
  reads is small compared with the blur itself, and depth compresses toward the centre.
* The optimum balances the two, and noise pushes it outward because it hurts the weak-signal side
  (section 3.4). A third, coarser pyramid scale does not help (slightly worse everywhere).
* **The scaling law holds exactly:** f/4 on 1.0-2.15 m and f/2 on 2.0-4.3 m have the same K*W and
  far/near ratio, and their error curves agree to the second decimal at every alpha.

So the 2.5 px figure belongs to the aperture shape and to FocalSplit's filters, not to the scene
range: it should be re-measured (cheaply, with the simulator) for a different estimator
configuration or a lens with a very different blur kernel.

### 3.4 Noise

Slide renders carry almost no noise (0.05 grey levels), a real sensor and the room renders a lot
more. With independent white noise added to each image (85 mm f/2, 2.0-4.3 m, centre -0.2 W, two
noise seeds per pair, which agree to 0.01 points):

| noise (grey levels, 0-255) | best alpha | best K*h (px) | best mean error | planes on range ends | 90 % sparse: best / ends |
|---|---|---|---|---|---|
| 0.05 (Blender slides) | 0.60 | 2.6 | 2.28 % | 3.52 % | 1.78 % / 2.82 % |
| 0.25 | 0.62 | 2.7 | 2.87 % | 3.90 % | 1.91 % / 2.89 % |
| 0.5 | 0.65 | 2.8 | 4.01 % | 4.76 % | 2.24 % / 3.07 % |
| 1 | 0.75 | 3.2 | 6.36 % | 6.67 % | 3.19 % / 3.65 % |
| 2 (about the room renders) | 0.87 | 3.7 | 10.39 % | 10.44 % | 5.05 % / 5.06 % |

(slide texture contrast: 9 grey levels standard deviation; refined optimum, 10 % band of alpha at
2 grey levels: 0.6-1.3.)

More noise moves the best separation outward (2.6 px -> 3.7 px at 2 grey levels) and flattens the
optimum: at low noise the right pair is worth a third of the error, at 1 grey level 5 %, at 2 grey
levels nothing. At room distance the whole DfDD signal is a fraction of a grey level; once noise
dominates, a wider separation (stronger signal) pays for its larger blur, and where exactly the
planes go stops mattering. The 90 %-sparse output is far more robust to noise than the dense map.
In practice: b = 2.5 px for a clean sensor (long exposure, frame averaging, or denoised pairs),
about 3.5 px for a noisy one, and when noise dominates, fix the noise before tuning the planes.

### 3.5 Rooms

The 16 room views of the 85 mm f/2 camera (two per Infinigen room, 4096 samples, as in
`indoor/camera_85mm_experiment.sh`) were rendered again with the focus at 2.3 / 4.0 m
(`focal_planes/render_rooms.sh`), and both pairs were run with their own slide calibration
(`focal_planes/room_check.py`; pixels whose true depth is inside 2.08-4.32 m; 15 views have enough
such pixels):

| pair (m) | input | dense MAE | 90 % sparse MAE |
|---|---|---|---|
| 1.984 / 4.416 | raw pair | 49.0 cm (14.7 %) | 23.8 cm (7.2 %) |
| 2.3 / 4.0 | raw pair | 49.4 cm (15.1 %), better in 6 of 15 views | 21.6 cm (6.5 %), better in 12 of 15 |
| 1.984 / 4.416 | denoised pair | 30.8 cm (9.0 %) | 17.0 cm (5.2 %) |
| 2.3 / 4.0 | denoised pair | **24.1 cm (7.5 %)**, better in 12 of 15 | 16.5 cm (4.9 %), better in 5 of 15 |

("denoised pair": the structured pair denoising of `monocular/pair_denoising.py`, applied to both
pairs before the unchanged model, as `run_experiment.py --pair-denoise` does.)

The room renders carry about 2 grey levels of render noise, and the raw result is what section 3.4
predicts for that noise: the dense map does not care where the planes are, because most room pixels
are too weakly textured for the defocus signal to rise above the noise, while the confident pixels
(the well-textured, high-signal ones, and the anchors of the monocular alignment) behave like the
slide and improve by 9 %. Denoising the pair lowers the effective noise everywhere, and then the
whole map behaves like the slide: the inward pair cuts the dense error by 22 % (30.8 -> 24.1 cm,
12 of 15 views). Over the four room comparisons the inward pair is better in three and unchanged
in one.

## 4. Rule

For FocalSplit as calibrated in this pipeline (2 pyramid scales, 320x240, natural-texture slide):

1. **Separation (in dioptres): `1/z_near - 1/z_far = 2 b / K`, with `b = 2.5 px` for a clean
   sensor and about 3.5 px for a noisy one (1-2 grey levels).** Equivalently: an object in focus on
   one sensor should be a blur disk about 2b = 5 px across on the other. This can be set on the
   bench without calibrating: focus one sensor on a target and move the other until the target's
   blur circle is about 5 px across. K = F^2 / (N * pixel pitch) in pixels per dioptre.
2. **Centre (in dioptres): between the geometric mean and the metric midpoint of the range.**
   The pair's dioptric centre at the metric midpoint, 1/c = (z_lo + z_hi)/2, minimises the mean
   relative error over depths spread evenly in metres; at the geometric mean, 1/c = sqrt(z_lo z_hi),
   it is more conservative (closer to the best worst-depth error). Both lie on the far side of the
   range's dioptric centre. Centre offsets searched per range (`refine_centres.py`, best alpha at
   each offset, offsets 0 to -0.4 W):

   | range (m) | far/near | best offset, mean error | best offset, worst depth | geometric-mean rule | metric-midpoint rule | mean error at best / at geometric rule |
   |---|---|---|---|---|---|---|
   | 2.5-3.5 | 1.40 | -0.2 | -0.3 | -0.08 | -0.17 | 1.01 % / 1.06 % |
   | 1.0-1.6 | 1.60 | -0.2 | -0.1 | -0.12 | -0.23 | 1.35 % / 1.39 % |
   | 1.2-2.0 | 1.67 | -0.3 | -0.1 | -0.13 | -0.25 | 1.42 % / 1.47 % |
   | 3-6 | 2.00 | -0.4 (edge) | -0.4 | -0.17 | -0.33 | 1.96 % / 2.20 % |
   | 2.0-4.3 | 2.15 | -0.4 (edge) | -0.3 | -0.19 | -0.36 | 2.10 % / 2.27 % |
   | 4-12 | 3.00 | -0.4 (edge) | -0.4 | -0.27 | -0.50 | 3.76 % / 4.20 % |
   | 1.5-6 | 4.00 | -0.4 (edge) | -0.2 | -0.33 | -0.60 | 7.40 % / 8.31 % |

   For ranges up to about 1.7:1 the choice hardly matters (0.05 points); for 2:1 and wider, centre
   near the metric midpoint.
3. **Check the range is reachable:** with this separation, the 10 % working range spans about
   +-6 px of blur around its centre (section 3.1). A range whose blur span K*W is much larger
   (1.5-6 m at f/2: 8 px) cannot be covered well by one pair; use two pairs or a smaller aperture.

`artifacts/focal_planes/blur_geometry.png` (`focal_planes/plot_blur_geometry.py`) shows the rule
on a dioptre axis: with the planes on the ends of 2.0-4.3 m an object on one focal plane is an
8.6 px disk in the other image; the rule's pair keeps it at 5 px.

Worked example, 85 mm f/2 room camera (K = 32.1 px/D), 2.0-4.3 m: h = 2.5/32.1 = 0.078 D.
Centre at the metric midpoint, 1/3.15 m = 0.317 D: focus at 1/(0.317 + 0.078) = 2.53 m and
1/(0.317 - 0.078) = 4.17 m, 2.13 % mean error and 5.1 % worst depth. Centre at the geometric
mean, 1/sqrt(2.0 * 4.3) = 0.341 D: 2.39 / 3.80 m, 2.31 % / 4.9 %. `design_focal_planes.py`,
which searches around the rule and then, among pairs within 5 % of the best mean error, picks the
one with the smallest worst-depth error, recommends 2.47 / 4.00 m: 2.18 % mean, 4.6 % worst,
against 4.30 % / 9.8 % for planes on the range ends.

Second example, the original close-range camera (35 mm f/1.4, 0.65-1.35 m, K*W = 3.1 px):
the rule gives 0.757 / 1.474 m (2.09 % mean, 4.4 % worst) and `design_focal_planes.py` recommends
0.774 / 1.542 m (2.03 %, 4.4 %), against 2.66 % / 5.7 % for planes on the range ends. The
separation hardly changes (K*h = 2.5 px against the range's own 3.1 px), but the far-shifted
centre moves the near plane inward and the far plane outward, beyond the range. So "move the
planes inward" is not the general answer; "set the separation by blur, then shift the pair toward
the far end" is.

In the language of the original question: for this camera the near plane goes 0.4-0.5 m inside the
near end and the far plane 0.15-0.5 m inside the far end (rule: 2.39-2.53 / 3.80-4.17 m; tool:
2.47 / 4.00 m; best worst-depth pair of the grid: 2.46 / 4.07 m; the error surface is flat between
them, and 2.3 / 4.0 m sits on it too).
That inward offset is a consequence of the camera, not a constant: at f/2.8 the same range wants
the planes on its ends, and at f/4 or smaller, outside them; in metres it also changes with the
range, because the rule lives in dioptres.

## 5. Caveats

* **Slides, not scenes.** Every number above is for the fronto-parallel fractal slide that fills
  the view. Real scenes add depth edges, slanted and textureless surfaces and texture whose scale
  changes with distance; they raise the error floor (section 3.5) and may move the optimum.
* **Ideal thin lens.** Blender's camera has a perfect disk aperture, no aberrations, no vignetting
  and no focus breathing. A real lens changes the blur kernel, and the 2.5 px figure depends on the
  kernel's shape (section 3.3). On hardware, confirm with calibrations at two or three separations
  around the predicted one before committing to a pair.
* **One estimator configuration.** The pixel figures belong to FocalSplit as configured here
  (2 pyramid scales, 5x5 pre-blur, 320x240). Another resolution or pyramid moves them; what stays
  is the method: express the separation in pixels of blur and search it with the simulator.
* **Calibration recipe.** Calibrated on the target range only, physics initialisation, 300 steps
  with a cosine-decayed learning rate. The pipeline's constant rate gives the same optimum but
  +-0.3 points of run-to-run scatter.
* **Noise model.** White Gaussian noise per image. Real sensors add shot noise that grows with
  brightness, fixed-pattern noise and demosaicing; use the measured noise level of the camera.
* **Metric.** Mean relative error over depths spaced uniformly in metres, and the worst depth. A
  different weighting (for example by how often each depth occurs in a scene) shifts the best centre
  more than the best separation.

## 6. How to run

```bash
# recommend a pair for a camera and a range (about 5 minutes; starts from the rule of thumb)
python dfdd_pipeline/focal_planes/design_focal_planes.py --lens-mm 85 --fstop 2 --z-lo 2.0 --z-hi 4.3
#   -> rule of thumb 2.53 / 4.17 m (2.13 %, worst 5.1 %), recommended 2.47 / 4.00 m (2.18 %, worst 4.6 %),
#      planes on the range ends 2.00 / 4.30 m (4.30 %, worst 9.8 %)
# add --noise <grey levels> --pair-correlation 0 for a camera with sensor noise

# once: sharp reference slide and validation renders (about 2 minutes)
blender --background --python dfdd_pipeline/focal_planes/render_reference_slide.py -- --output-dir artifacts/focal_planes/reference --device optix
python dfdd_pipeline/focal_planes/validate_simulator.py
python dfdd_pipeline/focal_planes/check_calibration.py --data artifacts/rtx3080_85mm_f2/data/zoom_range

# a focal-plane grid search for one camera and target range
python dfdd_pipeline/focal_planes/sweep_focal_planes.py --name f2_2to4p3 --lens-mm 85 --fstop 2 \
    --z-lo 2.0 --z-hi 4.3 --alphas 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.2 --deltas 0 -0.2
python dfdd_pipeline/focal_planes/analyze_sweep.py artifacts/focal_planes/sweeps/f2_2to4p3

# all scenarios and the cross-scenario pattern
bash dfdd_pipeline/focal_planes/run_scenarios.sh
python dfdd_pipeline/focal_planes/pattern_analysis.py

# confirm chosen pairs with real Blender slide sweeps
python dfdd_pipeline/focal_planes/blender_check.py --name f2_2p08to4p32 --pairs 2.24 3.50 2.3 4.0 \
    --existing 1.984 4.416 artifacts/rtx3080_85mm_f2/data/zoom_range
```
