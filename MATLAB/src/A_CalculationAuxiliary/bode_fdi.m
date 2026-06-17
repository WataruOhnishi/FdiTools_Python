function h = bode_fdi(sys, varargin)
%BODE_FDI - Bode plot of FRF(s) with optional uncertainty.
%   h = bode_fdi(Pest)
%   h = bode_fdi(Pest, 'unc', src, 'sigma', k, 'style', 'line'|'band', ...)
%   h = bode_fdi({sys1, sys2, ...}, ...)
%   h = bode_fdi(sys, [freq mag])          % positional uncertainty curve
%
% Plots the magnitude and phase of one frd / LTI model, or several overlaid,
% and optionally an uncertainty curve (e.g. the FRF standard deviation or
% noise level). Systems are drawn first, then the uncertainty, so a legend
% set by the caller as {sys1,...,sysN, uncertainty} always matches.
%
% INPUTS
%   sys   : an frd/tf/ss/zpk, or a cell array of them. Only the (1,1) channel
%           of a MIMO model is shown.
% NAME-VALUE OPTIONS
%   'unc'   : uncertainty source. One of
%               []                  none (default)
%               [freq(:) mag(:)]    explicit curve (Nx2, freq in 'unit')
%               mag(:)              magnitude on the first frd's frequencies
%               'fieldname'         a field of <sys>.UserData searched over the
%                                   given systems, e.g. 'sG','sCR','FRFn'
%             (A numeric value may also be given positionally as the 2nd arg.)
%   'sigma' : multiplier applied to the uncertainty magnitude (default 1)
%   'style' : 'line' (overlay 20log10(sigma*unc), default) or
%             'band' (shade |G_ref| +/- sigma*unc around the owning system)
%   'col'   : column of a multi-output UserData field to use (default 1)
%   'legend': cellstr of names; an (N+1)-th entry names the uncertainty curve
%   'unit'  : frequency-axis unit string for the label (default 'Hz')
%   'xlim'  : [fmin fmax] frequency limits (default: the frd frequency range)
%   'maglim': [lo hi] magnitude-axis limits (default: from the system curves,
%             so a wide confidence band does not flatten the plot)
%   'title' : title string for the magnitude axis
%   'legendloc' : legend location (default 'best'; e.g. 'southwest',
%                 'northeastoutside') - used only when 'legend' is given
%   'pmin','pmax' : phase wrap limits (default -180, 180)
%
% See also FRD, BODE, TIME2FRF_ML, TIME2FRF_LPM.
%   Author : Wataru Ohnishi, The University of Tokyo, 2026

% --- parse ---------------------------------------------------------------
unc = []; sigma = 1; style = 'line'; legNames = {}; unit = 'Hz';
xl = []; ttl = ''; pmin = -180; pmax = 180; col = 1; maglim = [];
legLoc = 'best';                         % legend location ('best','northeastoutside',...)
args = varargin;
if ~isempty(args) && ~(ischar(args{1}) || isstring(args{1}))
    unc = args{1}; args(1) = [];        % positional uncertainty
end
for k = 1:2:numel(args)
    key = lower(char(args{k})); val = args{k+1};
    switch key
        case 'unc',            unc = val;
        case 'sigma',          sigma = val;
        case 'style',          style = lower(char(val));
        case {'legend','names'},legNames = val;
        case 'unit',           unit = char(val);
        case 'xlim',           xl = val;
        case 'title',          ttl = val;
        case 'pmin',           pmin = val;
        case 'pmax',           pmax = val;
        case {'col','unccol'}, col = val;
        case {'maglim','ylim'},maglim = val;
        case {'legendloc','location'}, legLoc = char(val);
        otherwise
            error('bode_fdi:badopt','Unknown option "%s".',key);
    end
end
if ~iscell(sys), sys = {sys}; end
N = numel(sys);

% Frequency range: if not given, take it from the frd objects so the axis
% matches the measured band (and tf/ss models are sampled only there, not out
% to some arbitrary default).
if isempty(xl)
    ffrd = [];
    for k = 1:numel(sys)
        if isa(sys{k},'frd')
            fk = sys{k}.frequency(:);
            if strcmpi(sys{k}.FrequencyUnit,'rad/s'), fk = fk/(2*pi); end
            ffrd = [ffrd; fk(fk>0)]; %#ok<AGROW>
        end
    end
    if ~isempty(ffrd), fdef = [min(ffrd) max(ffrd)]; else, fdef = [1 1e4]; end
else
    fdef = xl;
end
fgrid = logspace(log10(fdef(1)), log10(fdef(2)), 600);

% --- figure & axes -------------------------------------------------------
h   = figure;
axM = subplot(2,1,1); set(axM,'XScale','log'); hold(axM,'on'); box(axM,'on');
axP = subplot(2,1,2); set(axP,'XScale','log'); hold(axP,'on'); box(axP,'on');

% --- systems (drawn first) ----------------------------------------------
magAll = [];                                 % system magnitudes (for the y-limits)
for k = 1:N
    [fHz, resp] = local_response(sys{k}, fgrid);
    nm = sprintf('G%d',k);
    if numel(legNames) >= k, nm = legNames{k}; end
    magdb = 20*log10(abs(resp));
    magAll = [magAll; magdb(isfinite(magdb))]; %#ok<AGROW>
    plot(axM, fHz, magdb, 'DisplayName', nm);
    plot(axP, fHz, local_wrapphase(rad2deg(angle(resp)),pmin,pmax), 'DisplayName', nm);
end

% --- uncertainty (drawn last) -------------------------------------------
[uf, um, uname, refsys] = resolve_unc(unc, sys, col);
% an (N+1)-th legend entry, if given, names the uncertainty curve
uncNamed = numel(legNames) >= N+1;
if uncNamed, uname = legNames{N+1}; end
if ~isempty(um)
    um = sigma*abs(um(:)); uf = uf(:);
    switch style
        case 'line'
            plot(axM, uf, 20*log10(um), '--', 'DisplayName', uname);
        case 'band'
            if isempty(refsys)
                warning('bode_fdi:noref', ...
                    'No reference frd for the band; drawing uncertainty as a line.');
                plot(axM, uf, 20*log10(um), '--', 'DisplayName', uname);
            else
                [rf, rresp] = local_response(refsys, uf');
                gmag = interp1(rf, abs(rresp), uf, 'linear', 'extrap');
                up = 20*log10(gmag + um);
                lo = 20*log10(max(gmag - um, eps));
                hb = fill(axM, [uf; flipud(uf)], [up; flipud(lo)], [0.6 0.6 0.6], ...
                    'EdgeColor','none','FaceAlpha',0.25, 'DisplayName', uname);
                if ~uncNamed, set(hb,'HandleVisibility','off'); end
            end
        otherwise
            error('bode_fdi:style','''style'' must be ''line'' or ''band''.');
    end
end

% --- cosmetics -----------------------------------------------------------
ylabel(axM,'Magnitude [dB]');
ylabel(axP,'Phase [deg]');
xlabel(axP, sprintf('Frequency [%s]', unit));
set(axP,'YTick', pmin:90:pmax); ylim(axP,[pmin pmax]);
% x-limits from the (frd) frequency range (or the user's 'xlim')
xlim(axM, fdef); xlim(axP, fdef);
% magnitude y-limits from the system responses, so a wide confidence band
% (whose lower edge can dive towards -inf) does not flatten the plot.
if ~isempty(maglim)
    ylim(axM, maglim);
elseif ~isempty(magAll)
    ylim(axM, [floor((min(magAll)-10)/10)*10, ceil((max(magAll)+10)/10)*10]);
end
if ~isempty(ttl), title(axM, ttl); end
if ~isempty(legNames), legend(axM,'show','Location',legLoc); end
end

% ===== local helpers ========================================================
function [fHz, resp] = local_response(sys, fgrid)
% Magnitude/phase samples of the (1,1) channel, frequency returned in Hz.
    sz = size(sys);
    if numel(sz) >= 2 && sz(1)*sz(2) > 1, sys = sys(1,1); end
    if isa(sys,'frd')
        fHz = sys.frequency(:);
        if strcmpi(sys.FrequencyUnit,'rad/s'), fHz = fHz/(2*pi); end
        resp = squeeze(sys.ResponseData); resp = resp(:);
    else
        fHz = fgrid(:);
        resp = squeeze(freqresp(sys, 2*pi*fHz)); resp = resp(:);
    end
end

function ph = local_wrapphase(ph, pmin, pmax)
    for kk = 1:numel(ph)
        while ph(kk) > pmax, ph(kk) = ph(kk) - 360; end
        while ph(kk) < pmin, ph(kk) = ph(kk) + 360; end
    end
end

function [uf, um, uname, refsys] = resolve_unc(unc, sys, col)
    uf = []; um = []; uname = 'uncertainty'; refsys = [];
    if isempty(unc), return; end
    if ischar(unc) || isstring(unc)
        field = char(unc); uname = field;
        for k = 1:numel(sys)
            ud = sys{k}.UserData;
            if isstruct(ud) && isfield(ud, field)
                refsys = sys{k};
                uf = local_freqHz(sys{k});
                v = ud.(field);
                um = v(:, min(col, size(v,2)));
                return
            end
        end
        warning('bode_fdi:nofield', ...
            'Uncertainty field "%s" not found in any system''s UserData.', field);
        uname = 'uncertainty';
    elseif isnumeric(unc)
        refsys = first_frd(sys);
        if size(unc,2) == 2
            uf = unc(:,1); um = unc(:,2);
        else
            um = unc(:);
            if ~isempty(refsys), uf = local_freqHz(refsys); end
        end
    end
end

function fHz = local_freqHz(sys)
    fHz = sys.frequency(:);
    if strcmpi(sys.FrequencyUnit,'rad/s'), fHz = fHz/(2*pi); end
end

function s = first_frd(sys)
    s = [];
    for k = 1:numel(sys)
        if isa(sys{k},'frd'), s = sys{k}; return; end
    end
end
