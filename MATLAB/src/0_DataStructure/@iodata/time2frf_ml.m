function Pest = time2frf_ml(dat, flagTime)
%TIME2FRF_ML - maximum-likelihood FRF estimation from iodata (SISO/SIMO/MIMO).
%   Pest = time2frf_ml(dat)
%   Pest = time2frf_ml(dat, flagTime)
%
% DAT  : periodic iodata whose UserData.ms holds the multisine used for
%        excitation. Three cases are handled automatically:
%   * SISO / SIMO (1 input, 1 experiment): delegates to the matrix-based
%     TIME2FRF_ML core; the result is identical to time2frf_ml(x,y,ms).
%     Statistics (sX2,sY2,cXY,sCR,sG,FRFn,...) are stored in UserData.
%
% CONVENTION: every estimator stores the FRF standard deviation in UserData.sG
%   - SISO/SIMO: (n_freq x n_out) ; MIMO: (n_out x n_in x n_freq). (The detailed
%   SISO statistics sCR/sX2/sY2/cXY/FRFn are also kept for the validation tests.)
%   * MIMO, multiple experiments (orthogonal/Hadamard multisine): with
%     n_exp >= n_inputs, the input matrix U(n_in x n_exp) is solved per
%     frequency, G = Y/U, giving the full FRF matrix.
%   * MIMO, single ZIPPERED experiment (each input owns disjoint excited
%     lines): at each excited line the active input is identified and the
%     corresponding FRF column is estimated; the other columns are NaN there.
%
% flagTime : if true, the time data is stored in Pest.UserData (SISO/SIMO only).
% Pest     : estimated FRF (frd). For MIMO, an (n_out x n_in) frd.
%
% See also IODATA, TIME2FRF_ML, TIME2FRF_LPM.
%   Author : Wataru Ohnishi, The University of Tokyo, 2026

if nargin < 2, flagTime = false; end
ne = nexp(dat);
nu = ninputs(dat);
ny = noutputs(dat);

% --- SISO / SIMO : reuse the matrix-based core ---------------------------
if ne == 1 && nu == 1
    if isempty(dat.UserData) || ~isfield(dat.UserData, 'ms')
        error('iodata:time2frf_ml:noms', ...
            ['UserData.ms (the multisine) is required. Build the data with ', ...
             'iodata(output, input, 1/ms.harm.fs, ''Period'', ms.nrofs, ', ...
             '''UserData'', struct(''ms'', ms)).']);
    end
    x  = dat.InputData;
    y  = dat.OutputData;
    ms = dat.UserData.ms;
    Pest = time2frf_ml(x, y, ms, flagTime);   % x double => global core
    return
end

% --- MIMO ---------------------------------------------------------------
if isempty(dat.UserData) || ~isfield(dat.UserData, 'ms')
    error('iodata:time2frf_ml:noms', 'UserData.ms (the multisine) is required.');
end
ms = dat.UserData.ms;
if iscell(ms), ms = ms{1}; end
fs    = 1/dat.Ts;
nrofs = dat.Period(1);
ex    = ms.ex(:);                       % excited bins (1-based into 1..nrofs/2)
freq  = (ex-1)*fs/nrofs;
nl    = numel(ex);

% period-averaged spectra at the excited lines, per experiment, with the
% per-component variance of the output mean spectrum (for the FRF uncertainty)
M    = floor(size(getexp(dat,1).OutputData,1)/nrofs);   % periods per experiment
Uall = zeros(nl, nu, ne);
Yall = zeros(nl, ny, ne);
sYa  = zeros(nl, ny, ne);
for e = 1:ne
    ed = getexp(dat, e);
    Uall(:,:,e)            = avgspec(ed.InputData,  nrofs, ex);
    [Yall(:,:,e), sYa(:,:,e)] = avgspec(ed.OutputData, nrofs, ex);
end

G  = zeros(ny, nu, nl);
sG = zeros(ny, nu, nl);                          % std of each FRF entry
if ne >= nu
    % multiple experiments: solve G(:,:,f) = Y/U at each frequency.
    % FRF std by propagating the output noise through W = pinv(U):
    %   sigma_G(o,i)^2 = sum_e sigma_Y(o,e)^2 |W(e,i)|^2
    for f = 1:nl
        Um = reshape(Uall(f,:,:), [nu, ne]);
        Ym = reshape(Yall(f,:,:), [ny, ne]);
        G(:,:,f) = Ym / Um;
        Wm = pinv(Um);                           % ne x nu
        sy = reshape(sYa(f,:,:), [ny, ne]);      % ny x ne
        for i = 1:nu, sG(:,i,f) = sqrt(sy*abs(Wm(:,i)).^2); end
    end
    method = 'orthogonal';
else
    % single zippered experiment: each input owns disjoint (interleaved) lines.
    % Estimate the active column (and its std) at each line, then interpolate
    % every column onto the full grid (per-channel resolution is 1/n_in).
    own = cell(nu,1);  Gsp = nan(ny,nu,nl);  Ssp = nan(ny,nu,nl);
    for f = 1:nl
        uvec = reshape(Uall(f,:,1), [nu, 1]);
        [~, ia] = max(abs(uvec));
        Gsp(:,ia,f) = reshape(Yall(f,:,1), [ny, 1]) / uvec(ia);
        Ssp(:,ia,f) = sqrt(reshape(sYa(f,:,1), [ny, 1])) / abs(uvec(ia));
        own{ia}(end+1) = f;
    end
    for i = 1:nu
        fi = own{i};
        for o = 1:ny
            G(o,i,:)  = interp1(freq(fi), squeeze(Gsp(o,i,fi)), freq, 'linear', 'extrap');
            sG(o,i,:) = interp1(freq(fi), squeeze(Ssp(o,i,fi)), freq, 'linear', 'extrap');
        end
    end
    method = 'zippered';
end

Pest = frd(G, freq, 'FrequencyUnit', 'Hz');
Pest.UserData.ms     = ms;
Pest.UserData.method = method;
Pest.UserData.sG     = sG;          % FRF standard deviation (ny x nu x nl)
Pest.UserData.nrofp  = M;           % periods per experiment (for FRFCONF)
end

% ===== local helper =========================================================
function [S, V] = avgspec(d, nrofs, ex)
% Period-averaged DFT of each column of d, sampled at the bins ex.
% S : mean spectrum ; V : per-component variance of the mean (across periods).
M  = floor(size(d,1)/nrofs);
nc = size(d,2);  ne = numel(ex);
S  = zeros(ne,nc);  V = zeros(ne,nc);
for c = 1:nc
    P = zeros(nrofs, M);
    for p = 1:M, P(:,p) = fft(d((p-1)*nrofs + (1:nrofs), c)); end
    Sp = P(ex,:);                       % ne x M  per-period spectra at the lines
    S(:,c) = mean(Sp,2);
    if M > 1, V(:,c) = var(Sp,0,2)/2/M; end   % var of the mean, per component
end
end
