function files = savefigs(name, varargin)
%SAVEFIGS - export all open figures to image files with a consistent name.
%   savefigs(name)
%   savefigs(name, 'Name',value, ...)
%   files = savefigs(...)
%
% Exports every open figure (in figure-number / creation order) to
%   <dir>/<name>_01.<ext>, <name>_02.<ext>, ...
% so the gallery markdown can reference them predictably.
%
% NAME : base file name (no extension). Tip: inside an example SCRIPT use
%        savefigs(mfilename) to auto-name by the script (works even after the
%        script's own 'clear all', because mfilename is a function, not a var).
%
% OPTIONS (Name,value)
%   'dir' : output folder (default: the 'plot' folder next to this file,
%           i.e. Examples/plot, created if missing)
%   'dpi' : resolution in dots-per-inch (default 150)
%   'ext' : image extension 'png'|'jpg'|'pdf'|... (default 'png')
%   'close' : true to close the figures after saving (default false)
%
% EXAMPLE
%   Step_MIMO2_NonparametricFRF      % run the example (creates figures)
%   savefigs('Step_MIMO2_NonparametricFRF')
%
% See also EXPORTGRAPHICS, PRINT.
%   Author : Wataru Ohnishi, The University of Tokyo, 2026

p = struct('dir','', 'dpi',150, 'ext','png', 'close',false);
for k = 1:2:numel(varargin), p.(lower(varargin{k})) = varargin{k+1}; end
if isempty(p.dir)
    p.dir = fullfile(fileparts(mfilename('fullpath')), 'plot');
end
if ~exist(p.dir,'dir'), mkdir(p.dir); end

figs = findobj(groot, 'Type','figure');
if isempty(figs)
    warning('savefigs:nofig','No open figures to save.');  files = {};  return
end
% order by figure number (creation order); figures without a number go last
nums = arrayfun(@(h) localnum(h), figs);
[~, ord] = sort(nums);
figs = figs(ord);

files = cell(numel(figs),1);
for i = 1:numel(figs)
    fn = fullfile(p.dir, sprintf('%s_%02d.%s', name, i, p.ext));
    try
        exportgraphics(figs(i), fn, 'Resolution', p.dpi);   % R2020a+
    catch
        print(figs(i), fn, ['-d' p.ext], sprintf('-r%d', p.dpi));
    end
    files{i} = fn;
    fprintf('saved %s\n', fn);
end
if p.close, close(figs); end
end

function n = localnum(h)
if isnumeric(h.Number) && ~isempty(h.Number), n = h.Number; else, n = inf; end
end
