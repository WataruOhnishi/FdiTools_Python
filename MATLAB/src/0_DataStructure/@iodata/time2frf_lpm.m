function Pest = time2frf_lpm(dat, varargin)
%TIME2FRF_LPM - Local Polynomial Method FRF estimation from iodata.
%   Pest = time2frf_lpm(dat)
%   Pest = time2frf_lpm(dat, 'order',R, 'halfwidth',n, 'band',[fl fh])
%   Pest = time2frf_lpm(dat, 'mode','broadband')   % force consecutive-bin mode
%
% DAT  : iodata holding the time-domain input/output. The data need NOT have
%        its transient removed - the LPM models it. Three cases are handled:
%   * SISO/SIMO (1 input, 1 experiment): periodic LPM if DAT is periodic and
%     carries a multisine (UserData.ms), else broadband.
%   * MIMO, multiple experiments (orthogonal multisine, n_exp>=n_in): the
%     transient-removed output spectrum Y0 is obtained per experiment with the
%     same local spike+polynomial fit as the SISO periodic LPM, then the full
%     FRF matrix is solved at each excited line, G = Y0 / U. This keeps the
%     FULL per-channel resolution (every excited line) AND removes the
%     transient, so a sharp resonance is resolved from a SHORT record.
%   * MIMO, single ZIPPERED experiment (n_in>1): each input owns disjoint
%     (interleaved) lines, so the SIMO periodic LPM is applied per input and
%     the columns are assembled (per-channel resolution = 1/n_in; a resonance
%     sharper than the per-channel line spacing cannot be resolved - use the
%     orthogonal multiple-experiment design above for sharp modes).
% Pest : estimated FRF (frd) with .sG, .T, .method='lpm', .ms in UserData.
%
% See also IODATA, TIME2FRF_LPM, TIME2FRF_ML.
%   Author : Wataru Ohnishi, The University of Tokyo, 2026

% strip 'mode' and detect broadband override; keep the rest for the core
forceBroadband = false;  keep = true(size(varargin));
for k = 1:2:numel(varargin)
    if ischar(varargin{k}) && strcmpi(varargin{k},'mode')
        forceBroadband = strcmpi(varargin{k+1},'broadband');  keep(k:k+1) = false;
    end
end
opts  = varargin(keep);
ne    = nexp(dat);
nu    = ninputs(dat);
ny    = noutputs(dat);
hasMs = ~isempty(dat.UserData) && isstruct(dat.UserData) && isfield(dat.UserData,'ms');

% =====================================================================
% MIMO, multiple experiments (orthogonal): per-experiment transient removal,
% then solve the full FRF matrix at each excited line.
% =====================================================================
if ne > 1
    if ~hasMs || isnan(dat.Period)
        error('iodata:time2frf_lpm:orthoperiodic', ...
            ['Multi-experiment LPM requires periodic data (Period set and ', ...
             'UserData.ms with the excited lines).']);
    end
    if ne < nu
        warning('iodata:time2frf_lpm:rank', ...
            'n_exp (%d) < n_in (%d): the input matrix is rank-deficient.', ne, nu);
    end
    [R, n, band] = parselpm(opts);
    ms = dat.UserData.ms;  if iscell(ms), ms = ms{1}; end
    nrofs = dat.Period(1);  ex = ms.ex(:);  nl = numel(ex);
    freq  = (ex-1)/nrofs / dat.Ts;                 % = (ex-1)*fs/nrofs
    Y0  = zeros(ny, ne, nl);                        % transient-removed spectra
    vY0 = zeros(ny, ne, nl);                        % their variances
    U0  = zeros(nu, ne, nl);
    powers = 0:R;
    for e = 1:ne
        ed = getexp(dat, e);
        P  = floor(size(ed.OutputData,1)/nrofs);
        if P < 2, error('iodata:time2frf_lpm:periods','need >= 2 periods.'); end
        nn = n;  if nn >= P, nn = P-1; end          % neighbours stay non-excited
        nw = 2*nn+1;  nprm = R+2;
        if nw < nprm
            error('iodata:time2frf_lpm:window', ...
                'Window too small: 2*halfwidth+1 (=%d) < order+2 (=%d).', nw, nprm);
        end
        PN = P*nrofs;
        Ye = fft(ed.OutputData(1:PN,:));            % PN x ny
        Ue = fft(ed.InputData(1:PN,:));             % PN x nu
        Kidx = P*(ex-1) + 1;                        % excited bins in the PN grid
        m  = (-nn:nn)';
        Kr = [double(m==0), m.^powers];            % nw x (R+2): spike + transient
        c11 = real(diag(inv(Kr'*Kr)));  q = nw - nprm;
        for j = 1:nl
            rr = min(max(Kidx(j)+m, 1), PN);
            Yw = Ye(rr,:);
            th = Kr\Yw;                            % (R+2) x ny
            Y0(:,e,j) = th(1,:).';                  % spike coeff = transient-free
            U0(:,e,j) = Ue(Kidx(j),:).';
            res = Yw - Kr*th;
            vY0(:,e,j) = (real(sum(conj(res).*res,1)).'/max(q,1)) * c11(1);
        end
    end
    G = zeros(ny,nu,nl);  sG = zeros(ny,nu,nl);
    for j = 1:nl
        Um = U0(:,:,j);  Ym = Y0(:,:,j);           % nu x ne , ny x ne
        G(:,:,j) = Ym / Um;
        W = pinv(Um);                              % ne x nu
        for i = 1:nu, sG(:,i,j) = sqrt(vY0(:,:,j) * abs(W(:,i)).^2); end
    end
    [freq, G, sG] = bandsel(freq, G, sG, band);
    Pest = frd(G, freq, 'FrequencyUnit','Hz');
    Pest.UserData.sG = sG;  Pest.UserData.method = 'lpm';
    Pest.UserData.ms = ms;  Pest.UserData.nrofp = ne;
    return
end

% single experiment from here on
u = dat.InputData;  y = dat.OutputData;
if isempty(u), error('iodata:time2frf_lpm:noinput','InputData is required.'); end
fs = 1/dat.Ts;

% =====================================================================
% MIMO, single ZIPPERED experiment: per-input SIMO LPM, then assemble.
% =====================================================================
if nu > 1
    if ~hasMs || isnan(dat.Period)
        error('iodata:time2frf_lpm:mimoperiodic', ...
            ['Zippered MIMO LPM requires periodic data (Period set and ', ...
             'UserData.ms with the excited lines).']);
    end
    ms = dat.UserData.ms;  if iscell(ms), ms = ms{1}; end
    nrofs = dat.Period(1);  ex = ms.ex(:);  nl = numel(ex);
    freq  = (ex-1)*fs/nrofs;
    P = floor(size(u,1)/nrofs);  Uavg = zeros(nl,nu);
    for i = 1:nu
        acc = zeros(nrofs,1);
        for p = 1:P, acc = acc + fft(u((p-1)*nrofs + (1:nrofs), i)); end
        Uavg(:,i) = acc(ex)/P;
    end
    [~, owner] = max(abs(Uavg), [], 2);
    G = nan(ny,nu,nl);  sGm = nan(ny,nu,nl);  Tm = nan(ny,nu,nl);
    for i = 1:nu
        idx = find(owner==i);  if isempty(idx), continue; end
        [FRFi,~,sGi,Ti] = time2frf_lpm(u(:,i), y, fs, opts{:}, ...
                                       'period', nrofs, 'lines', ex(idx));
        for o = 1:ny
            G(o,i,idx) = FRFi(:,o);  sGm(o,i,idx) = sGi(:,o);  Tm(o,i,idx) = Ti(:,o);
        end
    end
    for i = 1:nu
        fi = find(~isnan(squeeze(G(1,i,:))));
        for o = 1:ny
            G(o,i,:)   = interp1(freq(fi), squeeze(G(o,i,fi)),   freq, 'linear','extrap');
            sGm(o,i,:) = interp1(freq(fi), squeeze(sGm(o,i,fi)), freq, 'linear','extrap');
        end
    end
    Pest = frd(G, freq, 'FrequencyUnit','Hz');
    Pest.UserData.sG = sGm;  Pest.UserData.T = Tm;
    Pest.UserData.method = 'lpm';  Pest.UserData.ms = ms;
    return
end

% =====================================================================
% SISO / SIMO
% =====================================================================
if ~forceBroadband && ~isnan(dat.Period) && hasMs
    opts = [opts, {'period', dat.Period(1), 'lines', dat.UserData.ms.ex(:)}];
end
[FRF,freq,sG,T] = time2frf_lpm(u, y, fs, opts{:});

nrofo = size(FRF,2);
Pest = frd(FRF(:,1), freq, 'FrequencyUnit', 'Hz');
for o = 2:nrofo
    Pest = [Pest; frd(FRF(:,o), freq, 'FrequencyUnit', 'Hz')]; %#ok<AGROW>
end
Pest.UserData.sG = sG;  Pest.UserData.T = T;  Pest.UserData.method = 'lpm';
if hasMs, Pest.UserData.ms = dat.UserData.ms; end
end

% ===== local helpers ========================================================
function [R, n, band] = parselpm(opts)
R = 2;  n = 2;  band = [];
for k = 1:2:numel(opts)
    switch lower(opts{k})
        case 'order',                R = opts{k+1};
        case {'halfwidth','window'}, n = opts{k+1};
        case 'band',                 band = opts{k+1};
    end
end
end

function [f, G, sG] = bandsel(f, G, sG, band)
if isempty(band), return; end
sel = (f >= band(1)) & (f <= band(2));
f = f(sel);  G = G(:,:,sel);  sG = sG(:,:,sel);
end
