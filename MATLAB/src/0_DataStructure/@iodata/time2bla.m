function bla = time2bla(dat, M)
%TIME2BLA - robust MIMO Best Linear Approximation + nonlinear distortion level.
%   bla = time2bla(dat, M)
%
% Robust method (Pintelon-Schoukens): from M independent realizations of a
% random-phase multisine, the Best Linear Approximation (BLA) is the average
% of the per-realization FRFs. The scatter of the FRF ACROSS realizations
% contains both measurement noise AND the stochastic nonlinear distortions,
% while the scatter ACROSS PERIODS (within a realization) contains the noise
% only. Subtracting the two separates the nonlinear distortion level from the
% noise level.
%
% DAT : iodata holding n_exp = M * n_in experiments, ordered realization-major
%       (realization 1: its n_in orthogonal experiments, then realization 2,
%       ...). Each experiment is periodic with >= 2 periods and the data
%       carries UserData.ms (the excited lines, common to all realizations).
%       Different realizations use different random phases.
% M   : number of realizations.
%
% bla : struct with
%   .G        frd, the BLA (n_out x n_in), with stats echoed in its UserData
%   .freq     frequency [Hz]
%   .sG_total total std of the BLA  (noise + nonlinear distortions)
%   .sG_noise noise-only std of the BLA
%   .sG_nl    stochastic nonlinear distortion std  = sqrt(max(tot^2-noise^2,0))
%   .M, .nrofp
%
% See also IODATA, TIME2FRF_ML, TIME2NLD.
%   Author : Wataru Ohnishi, The University of Tokyo, 2026

ne = nexp(dat);
nu = ninputs(dat);
ny = noutputs(dat);
if isempty(dat.UserData) || ~isfield(dat.UserData,'ms')
    error('iodata:time2bla:noms','UserData.ms (the multisine) is required.');
end
if mod(ne, M) ~= 0
    error('iodata:time2bla:split', ...
        'n_exp (%d) must be a multiple of the number of realizations M (%d).', ne, M);
end
nexp_per = ne / M;                      % experiments per realization (= n_in)
ms = dat.UserData.ms;  if iscell(ms), ms = ms{1}; end

% --- per-realization FRF (reuse the orthogonal MIMO ML estimator) ----------
Gall = [];  Vn = [];  freq = [];  nrofp = NaN;
for m = 1:M
    idx  = (m-1)*nexp_per + (1:nexp_per);
    subY = dat.OutputData(idx);
    subU = dat.InputData(idx);
    sub  = iodata(subY, subU, dat.Ts, 'Period', dat.Period(1), ...
                  'UserData', struct('ms', ms));
    Pm = time2frf_ml(sub);
    if m == 1
        nl   = numel(Pm.Frequency);
        Gall = zeros(ny, nu, nl, M);
        Vn   = zeros(ny, nu, nl, M);
        freq = Pm.Frequency;
        nrofp = Pm.UserData.nrofp;
    end
    Gall(:,:,:,m) = Pm.ResponseData;
    Vn(:,:,:,m)   = Pm.UserData.sG.^2;          % per-realization noise variance
end

% --- combine: BLA and its variance decomposition ---------------------------
Gbla   = mean(Gall, 4);                          % BLA = average over realizations
Vtot   = var(Gall, 0, 4) / M;                    % var of the mean (noise + NL)
Vnoise = mean(Vn, 4) / M;                        % noise-only var of the mean
Vnl    = max(Vtot - Vnoise, 0);                  % stochastic nonlinear distortion

bla.G        = frd(Gbla, freq, 'FrequencyUnit', 'Hz');
bla.freq     = freq;
bla.sG_total = sqrt(Vtot);
bla.sG_noise = sqrt(Vnoise);
bla.sG_nl    = sqrt(Vnl);
bla.M        = M;
bla.nrofp    = nrofp;
bla.G.UserData.sG     = bla.sG_total;            % total std (for bode_fdi/frfconf)
bla.G.UserData.sG_nl  = bla.sG_nl;
bla.G.UserData.method = 'bla';
bla.G.UserData.nrofp  = nrofp;
end
