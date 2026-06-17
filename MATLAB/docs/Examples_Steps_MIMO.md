# FdiTools 3.0 — MIMO Step examples

Result gallery for the multi-input workflow (`Examples/Step_MIMO1` … `Step_MIMO6`),
on the shared 2×2 rank-one modal benchmark `mimobench` (modes ≈ 40 / 95 / 180 Hz).
See also [SISO Steps](Examples_Steps_SISO.md),
[SISO Tutorials](Examples_Tutorials_SISO.md), [MIMO Tutorial](Examples_Tutorials_MIMO.md).

---

## Step MIMO 1 — Excitation design (orthogonal & zippered)
Orthogonal (Hadamard) multisine over `n_in` experiments, plus the single-record
zippered design where each input owns interleaved excited lines.

![orthogonal multisine: time domain](../Examples/plot/Step_MIMO1_ExcitationDesign_01.png)
![orthogonal multisine: spectra](../Examples/plot/Step_MIMO1_ExcitationDesign_02.png)
![zippered multisine](../Examples/plot/Step_MIMO1_ExcitationDesign_03.png)
*The spectra are flat across the excited lines (the y-axis auto-zooms to
floating-point level because every line has identical amplitude).*

---

## Step MIMO 2 — Full 2×2 FRF (orthogonal & zippered) + confidence band
Orthogonal multi-experiment and single zippered estimates both recover the full
2×2 FRF (`time2frf_ml`); the per-entry 95% confidence band comes from `sG`.

![2×2 FRF: orthogonal vs zippered vs true](../Examples/plot/Step_MIMO2_NonparametricFRF_01.png)
![2×2 FRF with 95% confidence band](../Examples/plot/Step_MIMO2_NonparametricFRF_02.png)

---

## Step MIMO 3 — MIMO LPM (positioning)
Orthogonal, full-resolution MIMO LPM from short, transient-corrupted records.
The main resonances (incl. the 40 Hz peak in G₁₁) are captured, and the LPM
error matches settled-ML while ML-with-transient is far worse.

![experiment: start-up transient, shaded periods](../Examples/plot/Step_MIMO3_NonparametricFRF_LPM_positioning_01.png)
![2×2 FRF: LPM vs ML](../Examples/plot/Step_MIMO3_NonparametricFRF_LPM_positioning_02.png)
![2×2 FRF error vs true](../Examples/plot/Step_MIMO3_NonparametricFRF_LPM_positioning_03.png)

---

## Step MIMO 4 — Nonlinear distortions (robust BLA)
M independent random-phase realizations → Best Linear Approximation. The scatter
across realizations (total) minus the scatter across periods (noise) gives the
stochastic nonlinear distortion level, which here dominates the noise by
20–30 dB.

![BLA vs true linear plant](../Examples/plot/Step_MIMO4_NonlinearDistortions_01.png)
![noise vs nonlinear distortion levels](../Examples/plot/Step_MIMO4_NonlinearDistortions_02.png)
*Per entry: |BLA|, total std, nonlinear std and noise std — nonlinear ≫ noise.*

---

## Step MIMO 5 — Structured modal identification
`frf2modal` (rank-one residues, two-stage additive→modal, van der Hulst et al.
MSSP 2026) fits a modal model to the 2×2 FRF; it overlays the true plant and the
nonparametric FRF.

![2×2 structured modal identification](../Examples/plot/Step_MIMO5_ParametricEstimation_01.png)

---

## Step MIMO 6 — Model selection & validation
Sweep the number of flexible modes (the noise-normalized cost drops to its floor
at the true count of 3), then validate: the modeling error sits at the FRF
uncertainty σ_G across frequency.

![order selection: cost vs #modes](../Examples/plot/Step_MIMO6_SelectionValidation_01.png)
![selected modal model vs FRF](../Examples/plot/Step_MIMO6_SelectionValidation_02.png)
![residual validation: error vs uncertainty](../Examples/plot/Step_MIMO6_SelectionValidation_03.png)

```
--- identified modal parameters (selected order = 3) ---
 mode |  wn_true   wn_est [Hz] |  z_true    z_est
   1  |    40.00     40.00     |  0.010    0.010
   2  |    95.00     95.00     |  0.015    0.015
   3  |   180.00    179.98     |  0.020    0.020
modal-model FRF fit vs true : 98.52 %
noise-normalized cost       : 2.63
```
