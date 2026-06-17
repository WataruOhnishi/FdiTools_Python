function dat = pretreat(dat, varargin)
%PRETREAT - remove transients, offsets and trends from iodata. (MIMO)
%   dat = pretreat(dat)
%   dat = pretreat(dat, 'trans', nroft, 'trend', trend)
%   dat = pretreat(dat, nroft, trend)               % positional form
%
% DAT     : iodata with periodic measurement (Period must be set)
% trans   : number of leading transient periods to discard (default 1)
% trend   : flag for per-period trend removal {0,1} (default 0)
%
% The offset/transient/trend treatment is applied to every input and
% output channel of every experiment, reusing the matrix-based PRETREAT.
%
% See also IODATA, PRETREAT.
%   Author : Wataru Ohnishi, The University of Tokyo, 2026

if isnan(dat.Period)
    error('iodata:pretreat:noPeriod', ...
        'PRETREAT requires a periodic iodata (set the Period property).');
end

% --- argument parsing (name/value or positional) -------------------------
nroft = 1; trend = 0;
if numel(varargin) >= 1 && ischar(varargin{1})
    for k = 1:2:numel(varargin)
        switch lower(varargin{k})
            case {'trans','nroft'}, nroft = varargin{k+1};
            case 'trend',           trend = varargin{k+1};
            otherwise
                error('iodata:pretreat:badopt', ...
                    'Unknown option "%s".', varargin{k});
        end
    end
else
    if numel(varargin) >= 1, nroft = varargin{1}; end
    if numel(varargin) >= 2, trend = varargin{2}; end
end

nrofs = dat.Period(1);
fsamp = 1 / dat.Ts;

% --- apply per experiment, reusing the matrix-based pretreat -------------
ne = nexp(dat);
if ne == 1
    [dat.OutputData, ~] = pretreat(dat.OutputData, nrofs, fsamp, nroft, trend);
    if ~isempty(dat.InputData)
        [dat.InputData,  ~] = pretreat(dat.InputData,  nrofs, fsamp, nroft, trend);
    end
else
    yc = cell(1, ne); uc = cell(1, ne);
    for e = 1:ne
        [yc{e}, ~] = pretreat(dat.OutputData{e}, nrofs, fsamp, nroft, trend);
        if ~isempty(dat.InputData{e})
            [uc{e}, ~] = pretreat(dat.InputData{e}, nrofs, fsamp, nroft, trend);
        else
            uc{e} = [];
        end
    end
    dat.OutputData = yc;
    dat.InputData  = uc;
end
end
