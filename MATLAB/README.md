FdiTools
========

Frequency Domain System Identification MATLAB Toolbox.

> **v3.0** — the continued, major-upgraded version of
> [HoriFujimotoLab/FdiTools](https://github.com/HoriFujimotoLab/FdiTools) (v1–v2.1.1).
> The original repository remains available there; this version adds the
> `iodata` container, the Local Polynomial Method, and MIMO FRF/LPM/BLA.
> See **What's new in 3.0** below.

Main references:<br>
- R. Pintelon and J. Schoukens, *System Identification: A Frequency Domain Approach*, 2nd ed., Wiley-IEEE Press, 2012.
- J. Schoukens, R. Pintelon, Y. Rolain, *Mastering System Identification in 100 Exercises*, Wiley-IEEE Press, 2012.

## What's new in 3.0
- **`iodata`** — an `iddata`-compatible time-domain data container that works **without** the System Identification Toolbox (`toIddata`/`fromIddata` convert when it is available). Unifies the time-domain side of the whole pipeline.
- **Local Polynomial Method (`time2frf_lpm`)** — models the transient (leakage), so the FRF is identified from short, transient-corrupted records without discarding periods. Two modes: *periodic* (DFT of the full P-period record, so the non-excited spectral lines carry the transient) and *broadband* (arbitrary records).
- **MIMO FRF estimation (`time2frf_ml`)** — full transfer-matrix FRF from **orthogonal multiple-experiment** or **zippered single-experiment** multisines.
- **`bode_fdi`** redesigned + **`frfconf`** — FRF Bode plots with uncertainty (1σ line or shaded band) and confidence-bound radius (PS2012 eq. 2-40).

## Installation
```matlab
addpath(genpath('src'))     % adds src and all sub-packages (incl. @iodata)
```
### Required toolboxes
* MATLAB
* Control System Toolbox
### Optional
* System Identification Toolbox — only for `iodata.toIddata` / `iodata.fromIddata`
  (everything else runs without it)
* Signal Processing Toolbox — only for the legacy windowed estimators
  (`time2frf_h1`, `time2frf_log`, which use `hanning`/`bartlett`) and the
  `tfestimate` comparison in `Tutorial_1`. The main pipeline (multisine →
  `time2frf_ml`/`time2frf_lpm` → parametric estimation → validation) runs
  without it; `residtest` uses an FFT-based autocorrelation instead of `xcorr`.

# Overview

## Data structure
* `iodata` — `iddata`-compatible container: `OutputData`, `InputData`, `Ts`,
  `Period`, channel names, multiple experiments (cell), `UserData`.
  ```matlab
  dat  = iodata(output, input, 1/fs, 'Period', nrofs, 'UserData', struct('ms', ms));
  dat  = pretreat(dat, 'trans', 1, 'trend', 0);
  Pest = time2frf_ml(dat);          % FRF as an frd
  id   = toIddata(dat);             % -> iddata (needs SI Toolbox)
  ```

## ExcitationDesign
* Multisine (linear / quasi-logarithmic grid, full/odd/odd-odd, MIMO orthogonal)
* Chirp / swept sine, PRBS

## NonparametricFRF
* Periodic ML estimate `time2frf_ml`: $\hat G_{ML}(j\omega)$, sample (co)variances
  $\sigma_U^2,\sigma_Y^2,\sigma_{YU}^2$, FRF standard deviation $\sigma_{\hat G}$
  (`sG`; per-component `sCR`) and noise model `FRFn`. Every FRF estimator
  (`time2frf_ml`/`time2frf_lpm`/`time2bla`) exposes the FRF std as `UserData.sG`.
* **Local Polynomial Method** `time2frf_lpm`: transient handling for short records.
* **MIMO**: orthogonal (multi-experiment) or zippered (single-experiment).
* Uncertainty / confidence bounds: `bode_fdi`, `frfconf`.

## ParametricEstimation
* Deterministic: least squares, weighted LS, nonlinear LS (`lsfdi`, `wlsfdi`, `nlsfdi`)
* Stochastic: maximum likelihood, bootstrapped/generalized TLS (`mlfdi`, `btlsfdi`, `gtlsfdi`)

## SelectionValidation
Three complementary tests (`residtest`, `costtest`, `chi2test`):
* **Residual whiteness** — the *shape* of the residual (is correlation inside the
  white-noise bounds?).
* **Residual cost** — the *level* of the residual (has it dropped to the noise
  floor? compares estimators).
* **χ² modeling error vs the CR bound** — is the model error below the
  measurement uncertainty σ_Ĝ at every frequency?

The CR bound here is the *measurement* uncertainty (a lower bound on the FRF
*estimate* variance), not the residual. How to read each figure is detailed in
the [SISO Steps gallery](docs/Examples_Steps_SISO.md) (Step 6).

## Examples
Numbered `Step_*` scripts form the canonical motor-bench pipeline; `Tutorial_*`
scripts are focused, self-contained demos.

**Result galleries** (figures from every example):
[SISO Steps](docs/Examples_Steps_SISO.md) ·
[MIMO Steps](docs/Examples_Steps_MIMO.md) ·
[SISO Tutorials](docs/Examples_Tutorials_SISO.md) ·
[MIMO Tutorial](docs/Examples_Tutorials_MIMO.md).
Regenerate the images by running `Examples/export_all_figs` (saves to
`Examples/plot/` via `savefigs`).

| Script | Topic |
|---|---|
| `Step_1_ExcitationDesign` | multisine / PRBS / swept-sine design |
| `Step_2_NonparametricFRF` | periodic ML FRF + uncertainty / confidence band |
| `Step_3_NonparametricFRF_LPM_thermal` | LPM on a slow furnace (heater→temp): experiment-time saving |
| `Step_3_NonparametricFRF_LPM_positioning` | LPM on the positioning-stage benchmark (force→velocity) |
| `Step_4_NonlinearDistortions` | even/odd nonlinear distortion detection |
| `Step_5_ParametricEstimation` | deterministic & stochastic parametric estimation |
| `Step_6_SelectionValidation` | residual / cost / chi-square validation |
| `Step_MIMO1_ExcitationDesign` | MIMO orthogonal multisine design |
| `Step_MIMO2_NonparametricFRF` | full 2×2 FRF (orthogonal **and** zippered) + confidence band |
| `Step_MIMO3_NonparametricFRF_LPM_positioning` | MIMO LPM (orthogonal, full-resolution) from short, transient records |
| `Step_MIMO4_NonlinearDistortions` | robust MIMO BLA: noise vs nonlinear distortion levels (`time2bla`) |
| `Step_MIMO5_ParametricEstimation` | MIMO structured modal identification (`frf2modal`) |
| `Step_MIMO6_SelectionValidation` | mode-count selection + modal-model residual validation |
| `Tutorial_1_*` | random / chirp / qlog excitation tutorials |
| `Tutorial_2_iterative` | iterative (inverse-S/N) experiment design |
| `Tutorial_3_nonlinear_*` | nonlinear distortion analysis |
| `Tutorial_4_MIMO` | MIMO FRF (orthogonal / zippered) + structured modal identification |

# Example plots
Two-mass system setup

<img src="Examples/plot/twomass.jpg?raw=true" width="400">

## ExcitationDesign
<img src="Examples/plot/1_Multisine.png?raw=true" width="600">

## NonparametricFRF
<img src="Examples/plot/2_FRFest.png?raw=true" width="600">

## NonlinearDistortions
<img src="Examples/plot/3_NL.png?raw=true" width="600">

## ParametricEstimation
<img src="Examples/plot/4_deterministic.png?raw=true" width="600">

<img src="Examples/plot/4_stochastic.png?raw=true" width="600">
