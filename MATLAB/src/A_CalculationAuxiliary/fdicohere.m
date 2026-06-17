function Pest = fdicohere(Pest, dat)
%FDICOHERE - periodic (ensemble) coherence for an frd from TIME2FRF_ML.
%   Pest = fdicohere(Pest, dat)
%
% Computes the ensemble coherence over the periods of DAT at the excited
% lines and stores it in Pest.UserData.cxy (size nl x ny x nu). For each
% output o and input i it evaluates, bin by bin from the per-period DFTs,
%   gamma^2 = |sum_p Y_{o,p} conj(U_{i,p})|^2
%             / ( (sum_p |U_{i,p}|^2) (sum_p |Y_{o,p}|^2) ).
% No Signal Processing Toolbox required (uses fft only).
%
% PEST : frd from time2frf_ml (carries UserData.ms with the excited lines ex).
% DAT  : the periodic iodata used for the estimate (Period set, >= 2 periods).
%
% See also TIME2FRF_ML, BODE_FDI.
%   Author : Wataru Ohnishi, The University of Tokyo, 2019 (rev. 2026)
%%%%
ms    = Pest.UserData.ms;
nrofs = ms.nrofs;
ex    = ms.ex(:);
nl    = numel(ex);

ed = getexp(dat, 1);                          % first (or only) experiment
U  = ed.InputData;   Y = ed.OutputData;
nu = size(U,2);      ny = size(Y,2);
M  = floor(size(U,1)/nrofs);                  % number of periods
if M < 2
    error('fdicohere:periods','need >= 2 periods for coherence.');
end

% per-period DFTs, kept only at the excited bins
Up = zeros(nl, M, nu);  Yp = zeros(nl, M, ny);
for p = 1:M
    idx = (p-1)*nrofs + (1:nrofs);
    Uf  = fft(U(idx,:));   Yf = fft(Y(idx,:));
    Up(:,p,:) = Uf(ex,:);  Yp(:,p,:) = Yf(ex,:);
end

cxy = zeros(nl, ny, nu);
for o = 1:ny
    Yo  = reshape(Yp(:,:,o), nl, M);          % nl x M
    Syy = sum(abs(Yo).^2, 2);
    for i = 1:nu
        Ui  = reshape(Up(:,:,i), nl, M);      % nl x M
        Suu = sum(abs(Ui).^2, 2);
        Suy = sum(Yo.*conj(Ui), 2);
        cxy(:,o,i) = abs(Suy).^2 ./ (Suu .* Syy);
    end
end

Pest.UserData.cxy = cxy;
end
