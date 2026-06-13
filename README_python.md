# FdiTools (Python)

Python port of **FdiTools**, the Frequency-Domain System Identification toolbox.
It mirrors the MATLAB API and algorithms on top of [`numpy`](https://numpy.org/),
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
next to the script, and open interactive windows (set `FDI_NOSHOW=1` to skip the
windows and only save PNGs).

### Step 1–5 workflow (uses the original measurement data)

Python ports of the MATLAB `Step_1`–`Step_5` examples, running on the original
`Examples/private/*.mat` data:

| script | MATLAB original | data used |
|---|---|---|
| `step1_excitation_design.py` | `Step_1_ExcitationDesign.m` | – (signal design only) |
| `step2_nonparametric_frf.py` | `Step_2_NonparametricFRF.m` | `MultisineTypeA.mat` |
| `step3_nonlinear_distortions.py` | `Step_3_NonlinearDistortions.m` | `MultisineTypeB.mat` |
| `step4_parametric_estimation.py` | `Step_4_ParametricEstimation.m` | `MultisineTypeA.mat` |
| `step5_selection_validation.py` | `Step_5_SelectionValidation.m` | `MultisineTypeA.mat` |

`MultisineTypeA/B.mat` are read directly with SciPy (see
[`examples/_data.py`](examples/_data.py)) — **no MATLAB required**.

```bash
python examples/step2_nonparametric_frf.py
```

### Tutorials (use the benchmark plant `mdl.Pv(1,1)`)

Python ports of the MATLAB `Tutorial_*` examples:

| script | MATLAB original | notes |
|---|---|---|
| `tutorial_1_random.py` | `Tutorial_1_random.m` | random noise, Welch FRF (SciPy) + NLS fit |
| `tutorial_1_chirp.py` | `Tutorial_1_chirp.m` | swept sine, periodic H1 + NLS fit |
| `tutorial_1_qlog.py` | `Tutorial_1_qlog.m` | quasi-log multisine, full estimator panel |
| `tutorial_2_iterative.py` | `Tutorial_2_iterative.m` | 3 experiments combined with `fcat_fdi`/`fdel_fdi` |
| `tutorial_3_nonlinear_in.py` | `Tutorial_3_nonlinear_in.m` | input polynomial non-linearity |
| `tutorial_3_nonlinear_out.py` | `Tutorial_3_nonlinear_out.m` | output-feedback non-linearity (the Simulink `model_nl_out.slx` reproduced as a state-space ODE) |

These need the benchmark plant.  Until you convert it (below) they fall back to a
synthetic stand-in plant and print a note.

### Benchmark model `20160829_ident.mat`

This file stores MATLAB *control objects* (`mdl.Pv` is a 2×1 `zpk`, `mdl.Pp` too)
that SciPy cannot read.  Convert it **once in MATLAB** to plain state-space data:

```matlab
>> cd Examples/private
>> convert_ident_to_python      % writes ident_python.mat
```

After that the tutorials automatically use the real `mdl.Pv(1,1)`.  You can also
load it directly:

```python
from examples._data import load_ident, benchmark_plant
P0 = load_ident("Pv", (0, 0))   # control.StateSpace, == MATLAB mdl.Pv(1,1)
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
