# FdiTools 3.0 — SISO Tutorials

Result gallery for the single-input tutorials. See also
[SISO Steps](Examples_Steps_SISO.md), [MIMO Steps](Examples_Steps_MIMO.md),
[MIMO Tutorial](Examples_Tutorials_MIMO.md).

---

## Tutorial 1 — chirp (swept-sine) excitation
Swept-sine time/frequency signal, coherence, and the estimated FRF vs the true
plant.

![swept-sine time & frequency](../Examples/plot/Tutorial_1_chirp_01.png)
![coherence](../Examples/plot/Tutorial_1_chirp_02.png)
![FRF vs true](../Examples/plot/Tutorial_1_chirp_03.png)

## Tutorial 1 — random excitation
Non-periodic random excitation: coherence (good in-band, drops at the
anti-resonance and out of band) and the estimated FRF.

![coherence](../Examples/plot/Tutorial_1_random_01.png)
![FRF vs true](../Examples/plot/Tutorial_1_random_02.png)

## Tutorial 1 — quasi-logarithmic multisine
Quasi-log grid multisine, the time-domain input/output, FRF with noise model,
the 95% confidence band, and the parametric estimators.

![multisine crest factors](../Examples/plot/Tutorial_1_qlog_01.png)
![excited-line spectra](../Examples/plot/Tutorial_1_qlog_02.png)
![input / output time data](../Examples/plot/Tutorial_1_qlog_03.png)
![FRF with noise model](../Examples/plot/Tutorial_1_qlog_04.png)
![FRF with 95% confidence band](../Examples/plot/Tutorial_1_qlog_05.png)
*The 95% band widens at high frequency where the SNR drops.*

![deterministic estimators](../Examples/plot/Tutorial_1_qlog_06.png)
![stochastic estimators](../Examples/plot/Tutorial_1_qlog_07.png)
![best estimator (BTLS)](../Examples/plot/Tutorial_1_qlog_08.png)

---

## Tutorial 2 — iterative experiment design (inverse S/N)
Three experiments — wideband, then refined via the inverse signal-to-noise
ratio, then concentrated at high frequency — are merged (`fcat_fdi`) into one
low-uncertainty FRF. The crest-factor optimiser converges to CF ≈ 1.69 / 1.95 /
2.22 for the three designs.

Excitation designs and per-band FRFs for the three experiments:

![exp 1 crest factors](../Examples/plot/Tutorial_2_iterative_01.png)
![exp 1 spectra](../Examples/plot/Tutorial_2_iterative_02.png)
![exp 1 time data](../Examples/plot/Tutorial_2_iterative_03.png)
![exp 1 FRF](../Examples/plot/Tutorial_2_iterative_04.png)
![exp 2 crest factors](../Examples/plot/Tutorial_2_iterative_05.png)
![exp 2 spectra](../Examples/plot/Tutorial_2_iterative_06.png)
![exp 2 time data](../Examples/plot/Tutorial_2_iterative_07.png)
![exp 2 FRF](../Examples/plot/Tutorial_2_iterative_08.png)
![exp 3 crest factors](../Examples/plot/Tutorial_2_iterative_09.png)
![exp 3 spectra](../Examples/plot/Tutorial_2_iterative_10.png)
![exp 3 time data](../Examples/plot/Tutorial_2_iterative_11.png)
![exp 3 FRF](../Examples/plot/Tutorial_2_iterative_12.png)

Single vs iterative experiment, and the resulting estimation error:

![single experiment FRF](../Examples/plot/Tutorial_2_iterative_13.png)
![estimation error: single vs iterative](../Examples/plot/Tutorial_2_iterative_14.png)
*Key result: the iterative design lowers the estimation error in the targeted
high-frequency range.*

Parametric estimation on the merged FRF:

![deterministic estimators](../Examples/plot/Tutorial_2_iterative_15.png)
![stochastic estimators](../Examples/plot/Tutorial_2_iterative_16.png)
![best estimator](../Examples/plot/Tutorial_2_iterative_17.png)
*(GTLS is a rough starting-value estimator and can degenerate; MLE/BTLS are the
reliable ones.)*

---

## Tutorial 3 — nonlinear distortions, input nonlinearity
Distortion analysis at increasing input amplitudes (and a linear reference).

![multisine crest factors](../Examples/plot/Tutorial_3_nonlinear_in_01.png)
![excited-line spectra](../Examples/plot/Tutorial_3_nonlinear_in_02.png)
![noise + NL, input amp 1](../Examples/plot/Tutorial_3_nonlinear_in_03.png)
![noise + NL, input amp 0.1](../Examples/plot/Tutorial_3_nonlinear_in_04.png)
![noise + NL, input amp 10](../Examples/plot/Tutorial_3_nonlinear_in_05.png)
![without noise and nl](../Examples/plot/Tutorial_3_nonlinear_in_06.png)

## Tutorial 3 — nonlinear distortions, output nonlinearity
Same sweep for an output (Wiener-type) nonlinearity.

![multisine crest factors](../Examples/plot/Tutorial_3_nonlinear_out_01.png)
![excited-line spectra](../Examples/plot/Tutorial_3_nonlinear_out_02.png)
![noise + NL, input amp 1](../Examples/plot/Tutorial_3_nonlinear_out_03.png)
![noise + NL, input amp 0.1](../Examples/plot/Tutorial_3_nonlinear_out_04.png)
![noise + NL, input amp 10](../Examples/plot/Tutorial_3_nonlinear_out_05.png)
![without noise and nl](../Examples/plot/Tutorial_3_nonlinear_out_06.png)
