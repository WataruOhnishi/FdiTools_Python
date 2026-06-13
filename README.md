# FdiTools (Python)

Python port of **FdiTools**, the Frequency-Domain System Identification toolbox —
a Python version of [HoriFujimotoLab/FdiTools](https://github.com/HoriFujimotoLab/FdiTools).
It mirrors the original MATLAB API and algorithms on top of [`numpy`](https://numpy.org/),
[`scipy`](https://scipy.org/) and [`python-control`](https://python-control.readthedocs.io/).

> Main reference: R. Pintelon and J. Schoukens, *System Identification: A
> Frequency Domain Approach*, 2nd ed. Wiley-IEEE Press, 2012.

## Installation

```bash
pip install numpy scipy control            # runtime dependencies
pip install matplotlib                      # optional, for bode_fdi plotting
```

Then install the package (from the repository root):

```bash
pip install -e .
```

Run the test suite:

```bash
pip install pytest
pytest
```

## Module map

| MATLAB folder | Python subpackage | Contents |
|---|---|---|
| `1_ExcitationDesign` | `fditools.excitation` | `multisine`, `sweptsine`, `prbs`, `multisine2hdr`, phase helpers |
| `2_NonparametricFRF` | `fditools.nonparametric` | `pretreat`, `time2frf_ml`, `time2frf_h1`, `time2frf_log`, `splinefit` |
| `3_NonlinearDistortions` | `fditools.nonlinear` | `time2bla`, `time2nld` |
| `4_ParametricEstimation` | `fditools.parametric` | `lsfdi`, `wlsfdi`, `nlsfdi`, `mlfdi`, `gtlsfdi`, `btlsfdi`, `ssfdi` |
| `5_SelectionValidation` | `fditools.validation` | `chi2test`, `costtest`, `residtest` |
| `A_CalculationAuxiliary` | `fditools.auxiliary` | `ba2theta`, `theta2ba`, `ba2hm`, `hm2ba`, `hfrf`, `cr_rao`, `f2t`, `t2f`, `dbm`, `phs`, `fdel_fdi`, `fcat_fdi`, `fdicohere`, `bode_fdi` |

All names are also re-exported at the top level, e.g. `import fditools as fdi; fdi.multisine(...)`.

## Repository layout

```
fditools/        Python package (the toolbox)
tests/           pytest test-suite
examples/        Python example scripts (Step 1–5 + tutorials)
docs/img/        figures shown in this README
pyproject.toml   Python packaging / dependencies
MATLAB/          the original MATLAB toolbox (kept for reference)
  src/             MATLAB source functions
  Contents.m       MATLAB toolbox contents
  Examples/        MATLAB example scripts + measurement data (private/*.mat)
  README.md        original MATLAB README
```

Python is the main implementation; everything MATLAB lives under `MATLAB/`.

## Quick start

```python
import numpy as np
import control
import fditools as fdi

# true plant (for the demo)
P0 = control.tf([(2*np.pi*120)**2], [1, 2*0.02*2*np.pi*120, (2*np.pi*120)**2])

# 1) design a quasi-log multisine
harm = dict(fs=2000.0, df=1.0, fl=5.0, fh=400.0, fr=1.02)
options = dict(itp="r", ctp="c", dtp="f", gtp="q")
ms = fdi.multisine(harm, control.tf([1], [1]), options)

# experiment: 6 periods through the plant
u = np.tile(np.squeeze(ms.x[0, 0, :]), 6)
T = np.arange(u.size) / harm["fs"]
y = control.forced_response(P0, T, u).outputs

# 2) non-parametric FRF (maximum likelihood)
xp, _ = fdi.pretreat(u, ms.nrofs, harm["fs"], 1, 0)
yp, _ = fdi.pretreat(y, ms.nrofs, harm["fs"], 1, 0)
Pest = fdi.time2frf_ml(xp, yp, ms)          # -> fditools.FrfData

# 3) parametric estimation
n, mh, ml = 2, 0, 0
Hml, Hls = fdi.mlfdi(Pest, n, mh, ml, 500, 1e-10, 0, "c")
Hbtls, Hgtls = fdi.btlsfdi(Pest, n, mh, ml, 1.0, 500, 1e-10, "c")
sys_ml = control.tf(Hml[0, 0])              # a control.TransferFunction
```

A runnable end-to-end demo (with optional plots) is in
[`examples/tutorial_1_qlog.py`](examples/tutorial_1_qlog.py).

## Examples

All example scripts live in [`examples/`](examples/), save their figures as PNGs
next to the script, and open interactive windows (set `FDI_NOSHOW=1` to only save
PNGs).  They are ports of the original MATLAB scripts under
[`MATLAB/Examples/`](MATLAB/Examples).

```bash
python examples/step2_nonparametric_frf.py      # run any example
```

### Step 1–5 workflow

Ported from MATLAB `Step_1`–`Step_5`, running on the original motor-bench
measurement data (`MATLAB/Examples/private/*.mat`, read directly with SciPy —
**no MATLAB required**).

**Step 1 — excitation design** &nbsp;(`step1_excitation_design.py` ← `Step_1_ExcitationDesign.m`): multisine / PRBS / swept-sine.

![Step 1 multisine](docs/img/step1_multisine.png)

**Step 2 — non-parametric FRF** &nbsp;(`step2_nonparametric_frf.py` ← `Step_2_NonparametricFRF.m`): maximum-likelihood FRF, motor- and load-side (`MultisineTypeA.mat`).

![Step 2 motor](docs/img/step2_frf_motor.png)
![Step 2 load](docs/img/step2_frf_load.png)

**Step 3 — non-linear distortions** &nbsp;(`step3_nonlinear_distortions.py` ← `Step_3_NonlinearDistortions.m`): linear / even / odd / noise split (`MultisineTypeB.mat`).

![Step 3](docs/img/step3_nonlinear.png)

**Step 4 — parametric estimation** &nbsp;(`step4_parametric_estimation.py` ← `Step_4_ParametricEstimation.m`): SIMO deterministic (WLS/NLS) and stochastic (ML/BTLS) estimators.

![Step 4 deterministic](docs/img/step4_deterministic.png)
![Step 4 stochastic](docs/img/step4_stochastic.png)

**Step 5 — selection & validation** &nbsp;(`step5_selection_validation.py` ← `Step_5_SelectionValidation.m`): residual-whiteness, cost-function and chi-squared tests.

![Step 5 residuals](docs/img/step5_residuals.png)
![Step 5 cost](docs/img/step5_cost.png)
![Step 5 chi2](docs/img/step5_chi2.png)

### Tutorials

Ported from MATLAB `Tutorial_*`; they identify the benchmark plant `mdl.Pv(1,1)`.
Until you convert it (see below) they fall back to a synthetic stand-in plant and
print a note.  (Figures below use the real benchmark model.)

**Tutorial 1 — random noise** &nbsp;(`tutorial_1_random.py`): Welch FRF (SciPy) + NLS fit.

![Tutorial 1 random](docs/img/tutorial_1_random.png)

**Tutorial 1 — swept sine** &nbsp;(`tutorial_1_chirp.py`): periodic H1 + NLS fit.

![Tutorial 1 chirp](docs/img/tutorial_1_chirp.png)

**Tutorial 1 — quasi-log multisine** &nbsp;(`tutorial_1_qlog.py`): full estimator panel.

![Tutorial 1 qlog](docs/img/tutorial_1_qlog.png)

**Tutorial 2 — iterative** &nbsp;(`tutorial_2_iterative.py`): three experiments combined with `fcat_fdi`/`fdel_fdi`.

![Tutorial 2 iterative](docs/img/tutorial_2_iterative.png)

**Tutorial 3 — input non-linearity** &nbsp;(`tutorial_3_nonlinear_in.py`).

![Tutorial 3 input](docs/img/tutorial_3_nonlinear_in.png)

**Tutorial 3 — output non-linearity** &nbsp;(`tutorial_3_nonlinear_out.py`): the Simulink `model_nl_out.slx` (output fed back through a polynomial) reproduced as a state-space ODE.

![Tutorial 3 output](docs/img/tutorial_3_nonlinear_out.png)

### Benchmark model `20160829_ident.mat`

This file stores MATLAB *control objects* (`mdl.Pv` is a 2×1 `zpk`, `mdl.Pp` too)
that SciPy cannot read.  Convert it **once in MATLAB** to plain state-space data:

```matlab
>> cd MATLAB/Examples/private
>> convert_ident_to_python      % writes ident_python.mat
```

After that the tutorials automatically use the real `mdl.Pv(1,1)`.  Load it
directly with:

```python
from examples._data import load_ident, benchmark_plant
P0 = load_ident("Pv", (0, 0))   # control.StateSpace == MATLAB mdl.Pv(1,1)
P0, label = benchmark_plant()   # real model if converted, else synthetic
```

## API conventions

* **Transfer-function arrays** (`Hm`) are 2-D `object` NumPy arrays of SISO
  `control.TransferFunction`, indexed `Hm[o, i]` (output, input), mirroring the
  MATLAB `Hm(o, i)`.
* **`FrfData`** replaces the MATLAB `frd` object enriched with `UserData`.
  `Pest.freq` (Hz), `Pest.response` (`(nrofo, nrofi, nroff)`), `Pest.userdata`
  (`.X`, `.Y`, `.sX2`, `.sY2`, `.cXY`, `.sCR`, `.sGhat`, `.FRFn`, `.ms` ...).
  Use `Pest.frf_columns()` for the `(nroff, nrofh)` matrix and `Pest.to_frd()`
  for a genuine `control.FrequencyResponseData`.
* **Dual calling convention**: the iterative estimators accept either a `FrfData`
  (structured) or the classical positional argument list, exactly like MATLAB.
* **Model orders** `mh`/`ml`: a scalar or flat list is one entry per transfer
  function; a `(nrofo, nrofi)` array follows the MATLAB column-major convention.
* **`SYS` for validation tests** is a `dict` mapping a model name to a transfer
  function array, replacing the MATLAB struct.

## Known limitations vs. the MATLAB toolbox

* **Random phases** (`randph`) use NumPy's Mersenne-Twister; designs are
  reproducible within Python but **not bit-identical** to MATLAB's `rng`.
* **`msinl2p`** ports the in-only crest-factor minimisation used by `multisine`;
  the additional-harmonic (snow, `Fa`) and input/output (`H`) branches are not
  ported and raise `NotImplementedError`.
* **`splinefit`** is a SciPy-backed least-squares spline fit; robustness
  iterations and derivative constraints of the original are not ported.
* **`ssfdi`** is a direct port of the MATLAB "work in progress" function (the
  interactive order prompt is replaced by a required `order` argument).
* **`gtlsfdi`/`btlsfdi`** faithfully reproduce the original `try chol(A)…catch`
  behaviour (the `catch` branch always runs) so results match MATLAB.
* **Plotting** (`bode_fdi`) requires `matplotlib` and is imported lazily.
* True MIMO (`nrofi > 1` *and* `nrofo > 1`) follows the original code's index
  conventions, which are exercised mainly for SISO/SIMO/MISO in the toolbox.

## License

Same as the original FdiTools project (see [LICENSE](LICENSE)).
