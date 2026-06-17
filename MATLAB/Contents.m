% Frequency Domain Identification Toolbox
% Version 3.0  15-Jun-2026
%
% Data Structure
%   iodata         - iddata-compatible time-domain data container.
%                     Works without the System Identification Toolbox;
%                     toIddata/fromIddata convert when it is available.
%
% Excitation Design
%   multisine       - Multisine excitation generation
%   msinl2p         - Lp-norm optimization of multisine phase
%   schroed         - Schoeder multisine phase design
%   randph          - Random multisine phase design
%   lpnorm          - Lp-norm vector calculation
%   lin2qlog        - Linear to quasi-logarithmic frequency grid
%   orthogonal      - Transformation matrix for orthogonal multisines
%   effval          - calculate the effective signal value
%   swept           - Swept-Sine excitation signal generation
%   prbs            - Pseudo-Random-Binary-Sequence signal generation
%     
% Nonparametric Estimation
%   pretreat        - transients, offsets and trends removal
%   splinefit       - intricate drift/trend removal by b-spline fit
%   time2frf_h1     - Classic least-squares estimation of frf (H1)
%                     recommended for synchronized arbitrary measurements
%   time2frf_ml     - Stochastic maximum-likelihood estimation of frf
%                     SISO/SIMO, and MIMO via orthogonal multiple-experiment
%                     or zippered single-experiment multisines (iodata)
%   time2frf_lpm    - Local Polynomial Method estimation of frf; models the
%                     transient (periodic: full P-period DFT; or broadband) so
%                     short, transient-corrupted records can be used. SISO/SIMO,
%                     and MIMO via orthogonal (full-res) or zippered multisines
%   time2frf_log    - Non-linear logaritmic estimation of frf (Hlog)
%                     for non-synchronised or missing data measurements
%
% Non-Linear Distortions
%   time2nld        - Detect the even/odd non-linear contributions
%                     with random odd-odd multisine measurements.
%   time2bla        - Best Linear Approximation from multiple random-phase
%                     multisine realizations; separates noise vs nonlinear
%                     distortion levels (SISO matrix core / MIMO via iodata).
%
% Parametric Estimation
%   lsfdi           - Linear Least Squares analytical estimator
%                     recommended for starting values calculations.
%   wlsfdi          - Weighted Least Squares analytical estimator
%                     recommended for improved starting values calc.
%   nlsfdi          - Non-linear Least Squares numerical estimator
%                     requires only measured input-output freq data.
%   mlfdi           - Maximum-Likelihood stochastic estimator
%                     requires measured non-parametric noise model.
%   gtlsfdi         - Generalized Total Least Squares estimator
%                     adviced for starting values calculations.
%   btlsfdi         - Bootstrapped Total Least Squares estimator
%                     numerical robust version of maximum-likelihood.
%   ssfdi           - Subspace Identification for State-Space models.
%   frf2modal       - Structured modal identification of MIMO systems from FRF
%                     data (rank-one residues; proportional & general-viscous
%                     damping; two-stage additive->modal, van der Hulst et
%                     al., MSSP 2026)
%
% Selection-Validation
%   chi2test        - Chi-Squares test for estimator/model set
%                     validation using Cramer-Rao lower bound.
%   costtest        - Maximum-likelihood cost function test for
%                     estimator/model set validation.
%   residtest       - Identification Residuals for validation
%
% Calculation Auxiliary
%   f2t             - Calculate time domain from fourrier coefficients
%   t2f             - Calculate frequency domain from time signals
%   dbm             - Calculate FRF magnitude in decibel (20*log10)
%   phs             - Calculate FRF phase in degree augmented with glith removal
%   theta2ba        - Transform parameter vector to rational polynomial
%   ba2theta        - Transform rational polynomial to parameter vector
%   cr_rao          - Cramer-Rao Lower Bound of parameter covariance matrix
%   frfconf         - confidence-radius factor for a measured FRF (PS2012 2-40)
%   bode_fdi        - Bode plot of FRF(s) with optional uncertainty (line/band)
%
% Author: Ir. Thomas Beauduin
% University of Tokyo, Hori-Fujimoto Lab