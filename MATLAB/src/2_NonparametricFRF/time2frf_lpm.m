function [FRF,freq,sG,T] = time2frf_lpm(u,y,fs,varargin)
%TIME2FRF_LPM - Local Polynomial Method FRF estimation (SISO/SIMO).
%   [FRF,freq,sG,T] = time2frf_lpm(u,y,fs)                              % broadband
%   [FRF,freq,sG,T] = time2frf_lpm(u,y,fs,'period',N,'lines',ex)        % periodic
%   [...] = time2frf_lpm(...,'order',R,'halfwidth',n,'band',[fl fh])
%
% The Local Polynomial Method (LPM) estimates the FRF together with the
% transient (leakage) term, so it gives low-bias FRF estimates even when a
% long start-up transient is present (no transient period need be discarded).
%
% PERIODIC mode ('period',N,'lines',ex)  -- the proper LPM for periodic data
%   (Pintelon-Schoukens; Schoukens et al.). The full record of P periods of N
%   samples is transformed with one PN-point DFT, P times finer than a single
%   period. The excited lines then sit at every P-th bin; the bins BETWEEN
%   them carry the transient only. Around each excited bin K the response is
%       Y(K+m) = Y0(k)*[m==0] + sum_{r=0}^R t_r m^r ,   m = -n..n  (n<P)
%   i.e. the transient is a local polynomial fitted on the non-excited
%   neighbours and Y0 = G(w_k) U(K) is the spike at the excited bin. Solving
%   the least squares per line gives Ghat = Y0 / U(K).  (Eqs. (8)-(17).)
%
% BROADBAND mode (default) -- consecutive-bin local polynomial on the whole
%   record DFT, for arbitrary / non-periodic excitation.
%
% u,y     : time-domain input (N x 1) and output (N x nrofo)
% fs      : sampling frequency [Hz]
% 'order'     : transient polynomial order R                    (default 2)
% 'halfwidth' : half window n (neighbours each side, n<P)       (default 2)
% 'band'      : [fl fh] frequency band to return                (default full)
% 'period'    : samples per period N (enables periodic mode)
% 'lines'     : excited bin indices ex (1-based into 1..N/2), periodic mode
% FRF,freq,sG,T : FRF, frequency [Hz], FRF std, transient term (t_0)
%
% Reference: R. Pintelon, K. Barbe, G. Vandersteen, J. Schoukens, "Improved
%   (non-)parametric identification of dynamic systems excited by periodic
%   signals", MSSP 25(7) 2683-2704 (2011); Pintelon & Schoukens 2012, Ch. 7.
% Author : Wataru Ohnishi, The University of Tokyo, 2026
%%%%%

% --- options -------------------------------------------------------------
R = 2; n = 2; band = []; Nper = []; lines = [];
for k = 1:2:numel(varargin)
    switch lower(varargin{k})
        case 'order',                 R = varargin{k+1};
        case {'halfwidth','window'},  n = varargin{k+1};
        case 'band',                  band = varargin{k+1};
        case 'period',                Nper = varargin{k+1};
        case 'lines',                 lines = varargin{k+1}(:);
        otherwise
            error('time2frf_lpm:badopt','Unknown option "%s".',varargin{k});
    end
end

[Ntot,nrofi]  = size(u);
[Ny,nrofo]    = size(y);
if Ny ~= Ntot, error('time2frf_lpm:size','u and y must have equal length.'); end
if nrofi ~= 1
    error('time2frf_lpm:mimo', ...
        'Only single-input (SISO/SIMO) data is supported (MIMO LPM is planned).');
end

periodic = ~isempty(Nper) && ~isempty(lines);

if periodic
    % ===== PERIODIC LPM (PN-point DFT, transient from non-excited bins) ====
    P  = floor(Ntot/Nper);                  % number of periods
    if P < 2, error('time2frf_lpm:periods','periodic LPM needs >= 2 periods.'); end
    PN = P*Nper;
    U  = fft(u(1:PN));                       % PN x 1   (energy only at excited bins)
    Y  = fft(y(1:PN,:));                     % PN x nrofo
    exb  = lines;                            % 1-based excited bins of the 1-period grid
    Kidx = P*(exb-1) + 1;                    % 1-based index of those lines in the PN grid
    freqAll = (exb-1)*fs/Nper;
    nl   = numel(exb);
    if n >= P, n = P-1; end                  % neighbours must stay non-excited
    nprm = R + 2;                            % unknowns: Y0 + t_0..t_R
    nw   = 2*n + 1;
    if nw < nprm
        error('time2frf_lpm:window', ...
            'Window too small: 2*halfwidth+1 (=%d) < order+2 (=%d).',nw,nprm);
    end
    powers = 0:R;
    FRF = zeros(nl,nrofo); T = zeros(nl,nrofo); sG = zeros(nl,nrofo);
    for idx = 1:nl
        K  = Kidx(idx);
        m  = (-n:n)';
        rr = min(max(K+m,1),PN);             % PN bin indices (clamped at edges)
        Kr = [double(m==0), m.^powers];      % nw x (R+2): [spike , transient poly]
        Yw = Y(rr,:);
        th = Kr\Yw;                          % (R+2) x nrofo
        FRF(idx,:) = th(1,:)/U(K);           % Y0/U(K)
        T(idx,:)   = th(2,:);                % t_0 (transient at the excited bin)
        q = nw - nprm;
        if q >= 1
            res = Yw - Kr*th;
            c11 = real(diag(inv(Kr'*Kr)));
            s2  = real(sum(conj(res).*res,1))/q;
            sG(idx,:) = sqrt(s2*c11(1))/abs(U(K));
        else
            sG(idx,:) = NaN;
        end
    end
else
    % ===== BROADBAND LPM (consecutive bins of the whole-record DFT) ========
    Mmax = floor(Ntot/2);
    U = fft(u); Y = fft(y);
    Uvec = U(2:Mmax+1);                      % skip DC; bin m -> freq m*fs/Ntot
    Yvec = Y(2:Mmax+1,:);
    freqAll = (1:Mmax)'*fs/Ntot;
    nprm = 2*(R+1); nw = 2*n+1;
    if nw <= nprm
        error('time2frf_lpm:window', ...
            'Window too small: need 2*halfwidth+1 > 2*(order+1).');
    end
    [FRF,T,sG] = lpm_core(Uvec, Yvec, R, n);
end

% --- band selection ------------------------------------------------------
if isempty(band)
    sel = true(size(freqAll));
else
    sel = (freqAll >= band(1)) & (freqAll <= band(2));
end
freq = freqAll(sel);
FRF  = FRF(sel,:);
T    = T(sel,:);
sG   = sG(sel,:);
end

% ===== local helper (broadband consecutive-bin LPM) =========================
function [G,T,sG] = lpm_core(U, Y, R, n)
% Local polynomial of order R, half-window n, over the consecutive grid 1..L,
% modelling Y(k+r) = (sum g_i r^i) U(k+r) + (sum t_i r^i).
L = size(U,1); nrofo = size(Y,2);
np = 2*(R+1);
powers = 0:R;
G = zeros(L,nrofo); T = zeros(L,nrofo); sG = zeros(L,nrofo);
for m = 1:L
    lo = m-n; hi = m+n;
    if lo < 1,  lo = 1;  hi = min(2*n+1,L);     end
    if hi > L,  hi = L;  lo = max(1,L-2*n);      end
    idx = (lo:hi)';
    r   = idx - m;
    Rp  = r.^powers;
    K   = [U(idx).*Rp, Rp];
    Yw  = Y(idx,:);
    th  = K\Yw;
    G(m,:) = th(1,:);
    T(m,:) = th(R+2,:);
    q = numel(idx) - np;
    if q >= 1
        c11 = real(diag(inv(K'*K)));
        res = Yw - K*th;
        s2  = real(sum(conj(res).*res,1))/q;
        sG(m,:) = sqrt(s2*c11(1));
    else
        sG(m,:) = NaN;
    end
end
end
