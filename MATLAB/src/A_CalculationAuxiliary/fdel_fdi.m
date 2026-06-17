function sysout = fdel_fdi(sys, fmin, fmax, vars)
% FdiTools version of fdel
% Wataru Ohnishi, The University of Tokyo, 2019
%%%%

nfreq = numel(sys.freq);                 % original number of frequency lines
if nargin < 4
    vars = fieldnames(sys.UserData);     % consider every UserData field
end

[~,kmin] = min(abs(sys.freq - fmin));
[~,kmax] = min(abs(sys.freq - fmax));

sysout = fdel(sys,sys.freq(kmin:kmax));
sysout = fdel_fdi_UserData(sysout, vars, kmin, kmax, nfreq);

end

function out = fdel_fdi_UserData(in, sname, kmin, kmax, nfreq)
% Delete the [kmin:kmax] rows ONLY from frequency-indexed fields (rows ==
% nfreq). Non-frequency fields (ms struct, nrofp scalar, method char, time
% data x/y, ...) are left untouched - they are not frequency dependent.
for k = 1:numel(sname)
    f = sname{k};
    if ~isfield(in.UserData, f), continue; end
    temp = in.UserData.(f);
    if isnumeric(temp) && size(temp,1) == nfreq
        in.UserData.(f) = [temp(1:kmin-1,:); temp(kmax+1:end,:)];
    end
end
out = in;
end

