classdef iodata
%IODATA - Time-domain data container compatible with MATLAB iddata.
%
%   dat = iodata(y, u, Ts)
%   dat = iodata(y, u, Ts, 'PropertyName', value, ...)
%
% IODATA stores input/output measurement data together with the metadata
% required by the frequency-domain identification tools (sampling time,
% excitation period, channel names, multisine information, ...).
%
% The property names mirror MATLAB's System Identification Toolbox iddata
% object, so that code reads the same for both types and the two can be
% converted into each other. Unlike iddata, IODATA has NO dependency on
% the System Identification Toolbox: every FdiTools workflow runs with the
% Control System Toolbox alone. The optional converters toIddata /
% fromIddata are the only methods that require the toolbox.
%
% PROPERTIES (iddata compatible)
%   OutputData     : output samples, N-by-Ny (cell of such for multiple experiments)
%   InputData      : input samples,  N-by-Nu (cell, idem). [] for output-only data
%   Ts             : sampling time [s] (Ts = 1/fs)
%   Tstart         : start time of the record
%   Period         : number of samples in one excitation period (NaN if aperiodic)
%   Domain         : 'Time' (frequency-domain data is held in frd objects)
%   TimeUnit       : time unit string, default 'seconds'
%   InputName/OutputName : channel name cellstrings
%   InputUnit/OutputUnit : channel unit cellstrings
%   ExperimentName : experiment name cellstring (for multiple experiments)
%   Name, Notes    : free-form description
%   UserData       : FdiTools metadata struct (e.g. .ms multisine, .ex excited lines)
%
% See also IDDATA, PRETREAT, TIME2FRF_ML.
%   Author : Wataru Ohnishi, The University of Tokyo, 2026

    properties
        OutputData = []
        InputData  = []
        Ts = 1
        Tstart = []
        Period = NaN
        Domain = 'Time'
        TimeUnit = 'seconds'
        OutputName = {}
        InputName  = {}
        OutputUnit = {}
        InputUnit  = {}
        ExperimentName = {}
        Name = ''
        Notes = {}
        UserData = []
    end

    methods
        function dat = iodata(varargin)
            if nargin == 0
                return                                  % empty object
            end
            % If a single iddata is passed, convert it.
            if nargin == 1 && isa(varargin{1}, 'iddata')
                dat = iodata.fromIddata(varargin{1});
                return
            end
            dat.OutputData = varargin{1};
            if nargin >= 2, dat.InputData = varargin{2}; end
            if nargin >= 3 && ~isempty(varargin{3}), dat.Ts = varargin{3}; end

            % Name/value pairs
            if nargin > 3
                pv = varargin(4:end);
                if mod(numel(pv), 2) ~= 0
                    error('iodata:pairs', ...
                        'Property/value arguments must come in pairs.');
                end
                valid = properties(dat);
                for k = 1:2:numel(pv)
                    name = pv{k};
                    idx = find(strcmpi(name, valid), 1);
                    if isempty(idx)
                        error('iodata:badprop', ...
                            'Unknown iodata property "%s".', name);
                    end
                    dat.(valid{idx}) = pv{k+1};
                end
            end
            dat = validate(dat);
        end

        % ----- dimension helpers -------------------------------------------
        function n = nexp(dat)
            %NEXP Number of experiments held in DAT.
            if iscell(dat.OutputData)
                n = numel(dat.OutputData);
            else
                n = 1;
            end
        end

        function n = nsamples(dat)
            %NSAMPLES Samples per experiment (row vector for multiple experiments).
            if iscell(dat.OutputData)
                n = cellfun(@(c) size(c,1), dat.OutputData);
            else
                n = size(dat.OutputData, 1);
            end
        end

        function n = noutputs(dat)
            %NOUTPUTS Number of output channels.
            od = firstexp(dat.OutputData);
            n = size(od, 2);
        end

        function n = ninputs(dat)
            %NINPUTS Number of input channels.
            id = firstexp(dat.InputData);
            n = size(id, 2);
        end

        function n = nperiods(dat)
            %NPERIODS Number of whole excitation periods per experiment.
            if isnan(dat.Period)
                n = NaN;
            else
                n = floor(nsamples(dat) / dat.Period(1));
            end
        end

        function f = fs(dat)
            %FS Sampling frequency [Hz] (= 1/Ts).
            f = 1 / dat.Ts;
        end

        function varargout = size(dat, dim)
            %SIZE Data dimensions [Ns Ny Nu (Ne)].
            s = [nsamples(dat), noutputs(dat), ninputs(dat)];
            if nexp(dat) > 1
                s = [s, nexp(dat)];
            end
            if nargin == 2
                varargout{1} = s(dim);
            elseif nargout <= 1
                varargout{1} = s;
            else
                for k = 1:nargout, varargout{k} = s(min(k, numel(s))); end
            end
        end

        % ----- experiment handling -----------------------------------------
        function out = getexp(dat, k)
            %GETEXP Extract experiment K as a single-experiment iodata.
            if nexp(dat) == 1
                if k ~= 1, error('iodata:getexp', 'Only one experiment.'); end
                out = dat; return
            end
            out = dat;
            out.OutputData = dat.OutputData{k};
            if iscell(dat.InputData) && ~isempty(dat.InputData)
                out.InputData = dat.InputData{k};
            end
            if iscell(dat.ExperimentName) && numel(dat.ExperimentName) >= k
                out.ExperimentName = dat.ExperimentName(k);
            end
        end

        function dat = merge(varargin)
            %MERGE Combine iodata objects into one multi-experiment object.
            dats = varargin;
            ref = dats{1};
            yc = {}; uc = {}; en = {};
            for d = 1:numel(dats)
                obj = dats{d};
                for e = 1:nexp(obj)
                    ex = getexp(obj, e);
                    yc{end+1} = ex.OutputData;       %#ok<AGROW>
                    uc{end+1} = ex.InputData;        %#ok<AGROW>
                    if ~isempty(ex.ExperimentName)
                        en{end+1} = ex.ExperimentName{1}; %#ok<AGROW>
                    else
                        en{end+1} = sprintf('Exp%d', numel(yc)); %#ok<AGROW>
                    end
                end
            end
            dat = ref;
            dat.OutputData = yc;
            dat.InputData  = uc;
            dat.ExperimentName = en;
            dat = validate(dat);
        end

        % ----- iddata interoperability (optional toolbox) ------------------
        function id = toIddata(dat)
            %TOIDDATA Convert to a System Identification Toolbox iddata object.
            assertIdentToolbox();
            % iddata accepts matrices (single experiment) or cell arrays
            % (multiple experiments) for the data arguments directly.
            id = iddata(dat.OutputData, dat.InputData, dat.Ts);
            if ~isnan(dat.Period), id.Period = dat.Period; end
            if ~isempty(dat.Tstart), id.Tstart = dat.Tstart; end
            id.TimeUnit = dat.TimeUnit;
            if ~isempty(dat.OutputName), id.OutputName = dat.OutputName; end
            if ~isempty(dat.InputName),  id.InputName  = dat.InputName;  end
            if ~isempty(dat.OutputUnit), id.OutputUnit = dat.OutputUnit; end
            if ~isempty(dat.InputUnit),  id.InputUnit  = dat.InputUnit;  end
            if ~isempty(dat.ExperimentName), id.ExperimentName = dat.ExperimentName; end
            id.Name = dat.Name;
            id.Notes = dat.Notes;
            id.UserData = dat.UserData;
        end

        function disp(dat)
            ne = nexp(dat);
            if ne == 1
                fprintf('  iodata: %d samples, %d output(s), %d input(s)\n', ...
                    nsamples(dat), noutputs(dat), ninputs(dat));
            else
                fprintf('  iodata: %d experiments, %d output(s), %d input(s)\n', ...
                    ne, noutputs(dat), ninputs(dat));
            end
            fprintf('    Ts = %g s (fs = %g Hz)', dat.Ts, 1/dat.Ts);
            if ~isnan(dat.Period)
                fprintf(', Period = %d samples', dat.Period(1));
                np = nperiods(dat);
                fprintf(' (%s periods)', mat2str(np));
            end
            fprintf('\n');
        end
    end

    methods (Static)
        function dat = fromIddata(id)
            %FROMIDDATA Build an iodata from a System Identification iddata.
            assertIdentToolbox();
            y = id.OutputData; u = id.InputData;
            dat = iodata(y, u, id.Ts);
            try
                p = id.Period;
                if isfinite(p(1)), dat.Period = p(1); end
            catch
            end
            copy = {'Tstart','TimeUnit','OutputName','InputName', ...
                    'OutputUnit','InputUnit','ExperimentName', ...
                    'Name','Notes','UserData'};
            for c = 1:numel(copy)
                try
                    dat.(copy{c}) = id.(copy{c});
                catch
                end
            end
        end
    end

    methods (Access = private)
        function dat = validate(dat)
            % normalize empties and check consistency
            if iscell(dat.OutputData)
                no = numel(dat.OutputData);
                if isempty(dat.InputData)
                    dat.InputData = repmat({[]}, 1, no);
                elseif ~iscell(dat.InputData)
                    error('iodata:expmismatch', ...
                        'OutputData is multi-experiment but InputData is not.');
                elseif numel(dat.InputData) ~= no
                    error('iodata:expmismatch', ...
                        'OutputData and InputData have different experiment counts.');
                end
                for e = 1:no
                    checkrows(dat.OutputData{e}, dat.InputData{e}, e);
                end
            else
                checkrows(dat.OutputData, dat.InputData, 1);
            end
        end
    end
end

% ===== local helpers ========================================================
function a = firstexp(x)
    if iscell(x)
        if isempty(x), a = []; else, a = x{1}; end
    else
        a = x;
    end
end

function checkrows(y, u, e)
    if ~isempty(u) && ~isempty(y) && size(u,1) ~= size(y,1)
        error('iodata:rowmismatch', ...
            'Experiment %d: InputData and OutputData must have the same number of rows.', e);
    end
end

function assertIdentToolbox()
    % The conversion only needs to construct an iddata object, so the
    % presence of the iddata class on the path is the reliable test
    % (license('test',...) can give false negatives with network licenses).
    if exist('iddata', 'class') ~= 8
        error('iodata:noIdentToolbox', ...
            ['This conversion requires the System Identification Toolbox, ', ...
             'which is not available. All other FdiTools functionality ', ...
             'works without it.']);
    end
end
