%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CONVERT BENCHMARK MODEL FOR PYTHON
% ----------------------------------
% 20160829_ident.mat stores MATLAB Control System Toolbox objects (ss/tf)
% inside the struct 'mdl' (fields .Pv and .Pp).  Python's scipy.io cannot
% reconstruct those objects, so this script exports their plain state-space
% data (A,B,C,D,Ts) into 'ident_python.mat', which the Python loader
% examples/_data.py:load_ident() reads.
%
% Usage (in MATLAB, from this folder):
%   >> cd MATLAB/Examples/private
%   >> convert_ident_to_python
%
% Author: Wataru Ohnishi / FdiTools Python port
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; clc;

S = load('20160829_ident.mat');     % provides 'mdl' (and 'frf')
mdl = S.mdl;

models = struct();
channels = {'Pv', 'Pp'};
for c = 1:numel(channels)
    name = channels{c};
    if ~isfield(mdl, name) && ~isprop(mdl, name)
        continue;
    end
    G = ss(mdl.(name));             % force state-space form
    [no, ni] = size(G);            % number of outputs / inputs
    for o = 1:no
        for i = 1:ni
            g = ss(G(o, i));
            key = sprintf('%s_%d%d', name, o, i);   % e.g. Pv_11
            entry = struct();
            entry.A  = g.A;
            entry.B  = g.B;
            entry.C  = g.C;
            entry.D  = g.D;
            entry.Ts = g.Ts;        % 0 for continuous, >0 for discrete
            models.(key) = entry;
        end
    end
end

save('ident_python.mat', 'models', '-v7');
fprintf('wrote ident_python.mat with systems: %s\n', ...
        strjoin(fieldnames(models)', ', '));
