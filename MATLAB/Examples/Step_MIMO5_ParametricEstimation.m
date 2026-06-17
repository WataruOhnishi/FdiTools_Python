%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MIMO PARAMETRIC ESTIMATION (STRUCTURED MODAL):
% ----------------------------------------------
% Descr.:   Structured modal identification of the 2x2 MIMO benchmark from its
%           estimated FRF, with frf2modal (rank-one residues; two-stage
%           additive -> modal projection), following van der Hulst et al.,
%           MSSP 247 (2026) 113948. Recovers physical modal parameters
%           (frequency, damping, mode shapes).
% System:   2x2 rank-one modal benchmark (mimobench).
% Author:   Wataru Ohnishi, The University of Tokyo, 2026
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc;
[G0, wn_true, ze_true] = mimobench();

%% STEP 1: MIMO FRF (orthogonal multisine, as in Step_MIMO2)
harm.fs = 2500;  harm.df = 1;  harm.fl = 1;  harm.fh = 500;  harm.fr = 1.02;
options.itp = 'r'; options.ctp = 'c'; options.dtp = 'f'; options.gtp = 'l';
nrofi = 2;
ms    = multisine(harm, repmat(tf(1),[1,nrofi]), options);
nrofs = ms.nrofs;  nexp = nrofi;  nrofp = 3;  trans = 1;  nptot = nrofp + trans;
rng('default');  t = (0:nptot*nrofs-1)'/harm.fs;
uc = cell(1,nexp);  yc = cell(1,nexp);
for j = 1:nexp
    Uj  = squeeze(ms.x(:,j,:)).';
    in  = repmat(Uj, [nptot,1]);
    out = lsim(G0, in, t) + 1e-2*randn(nptot*nrofs, 2);
    uc{j} = in;  yc{j} = out;
end
dat  = iodata(yc, uc, 1/harm.fs, 'Period', nrofs, 'UserData', struct('ms',ms));
dat  = pretreat(dat, 'trans', trans);
Pest = time2frf_ml(dat);                % 2x2 frd

%% STEP 2: STRUCTURED MODAL IDENTIFICATION
nflex = numel(wn_true);
[modal, Pm] = frf2modal(Pest, 0, nflex, 'initfreq', wn_true*1.05);

fprintf('\n--- identified modal parameters ---\n');
fprintf(' mode |  wn_true   wn_est [Hz] |  z_true    z_est\n');
for i = 1:nflex
    fprintf('  %2d  | %8.2f  %8.2f    | %6.3f   %6.3f\n', ...
        i, wn_true(i), modal.wn(i), ze_true(i), modal.zeta(i));
end

%% STEP 3: VALIDATE THE MODAL MODEL AGAINST THE FRF
G0g = frd(freqresp(G0, 2*pi*Pest.Frequency), Pest.Frequency, 'FrequencyUnit','Hz');
Pmg = frd(freqresp(Pm, 2*pi*Pest.Frequency), Pest.Frequency, 'FrequencyUnit','Hz');
figure('Name','MIMO structured modal identification');
bode(G0g, Pest, Pmg); grid on;
legend('true','nonparametric FRF','identified modal model','Location','best');
title('2\times2 structured modal identification (frf2modal)');

g0 = G0g.ResponseData(:);  gm = Pmg.ResponseData(:);
fprintf('modal model FRF fit vs true : %.2f %%\n', 100*(1-norm(gm-g0)/norm(g0-mean(g0))));

% NOTE: frf2modal also supports general-viscous damping via 'damping','general'
% (complex mode shapes). The classical single-denominator estimators (mlfdi,
% nlsfdi, ...) target SISO/SIMO; frf2modal is the MIMO, modal-structured route.
