%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TUTORIAL OF MIMO FRF MEASUREMENT
% --------------------------------
% Descr.:   2-input / 2-output FRF identification on the shared MIMO benchmark
%           mimobench (the SAME 2x2 rank-one modal plant used by the Step_MIMO
%           series, modes ~40 / 95 / 180 Hz). Two excitation strategies are
%           demonstrated and compared against the true plant:
%             (A) orthogonal multisine over MULTIPLE experiments
%             (B) ZIPPERED multisine in a SINGLE experiment
%           The orthogonal FRF then feeds structured modal identification
%           (frf2modal), proportional and general-viscous damping.
% System:   2x2 rank-one modal benchmark (mimobench), shared with Step_MIMO*.
% Author:   Wataru Ohnishi, The University of Tokyo, 2026
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc;
[G0, wn_true, ze_true] = mimobench();        % true 2x2 plant (shared benchmark)

% Common experiment settings (aligned with the Step_MIMO series)
harm.fs = 2500;     harm.df = 1;             % sampling / resolution
harm.fl = 1;        harm.fh = 500;           % excited band (1 Hz - 500 Hz)
harm.fr = 1.02;                               % (qlog ratio, unused for 'l')
nrofp = 5;          trans = 1;                % periods kept / transient periods
nptot = nrofp + trans;
outnoise = 1e-3;                              % output noise (sets the FRF floor)

%% ===== METHOD A: ORTHOGONAL MULTISINE, MULTIPLE EXPERIMENTS =====
% multisine(nrofi=2) builds 2 orthogonal (Hadamard) experiments: in each
% experiment BOTH inputs are driven with phase patterns that make the input
% matrix U(2x2) invertible at every frequency, so G = Y/U gives the full 2x2.
options.itp = 'r'; options.ctp = 'c'; options.dtp = 'f'; options.gtp = 'l';
nrofi = 2;
Hampl = repmat(tf(1),[1,nrofi]);
ms    = multisine(harm, Hampl, options);
nrofs = ms.nrofs;
nexp  = nrofi;                              % orthogonal -> nrofi experiments

rng('default');
t  = (0:nptot*nrofs-1)'/harm.fs;
uc = cell(1,nexp); yc = cell(1,nexp);
for j = 1:nexp
    Uj  = squeeze(ms.x(:,j,:)).';          % one period, nrofs x 2 (experiment j)
    in  = repmat(Uj, [nptot,1]);
    out = lsim(G0, in, t);
    out = out + outnoise*randn(size(out));
    uc{j} = in;  yc{j} = out;
end
% multi-experiment iodata (cell per experiment); estimator auto-detects MIMO
datA = iodata(yc, uc, 1/harm.fs, 'Period', nrofs, 'UserData', struct('ms',ms));
datA = pretreat(datA, 'trans', trans);
Pa   = time2frf_ml(datA);                  % full 2x2 frd

figure('Name','MIMO FRF: orthogonal multiple experiments');
G0a = frd(freqresp(G0, 2*pi*Pa.Frequency), Pa.Frequency, 'FrequencyUnit','Hz');
bode(G0a, Pa); grid on;
legend('true','estimated (orthogonal, 2 exp)','Location','best');
title('2\times2 FRF - orthogonal multisine, multiple experiments');

%% ===== METHOD B: ZIPPERED MULTISINE, SINGLE EXPERIMENT =====
% One experiment, but each input excites DISJOINT (interleaved) lines:
% input 1 -> odd excited lines, input 2 -> even ones. At any excited line only
% one input is active, so each FRF column is read off directly. Full 2x2 from a
% single record, at the cost of halved per-channel frequency resolution.
nrofsZ = round(harm.fs/harm.df);
fbin   = (0:nrofsZ/2-1)*harm.df;           % freq of bin k (1-based) = (k-1)*df
inband = find(fbin >= harm.fl & fbin <= harm.fh);   % 1-based bin indices
ex1 = inband(1:2:end);  ex2 = inband(2:2:end);      % zipper the lines

U  = zeros(nrofsZ,2);
ph1 = exp(1i*2*pi*rand(numel(ex1),1));
ph2 = exp(1i*2*pi*rand(numel(ex2),1));
U(ex1,1) = ph1;  U(nrofsZ-ex1+2,1) = conj(ph1);     % conjugate-symmetric ->
U(ex2,2) = ph2;  U(nrofsZ-ex2+2,2) = conj(ph2);     % real time signals
uz = real(ifft(U));
uz = uz ./ max(abs(uz),[],1);              % normalize each input amplitude

inz  = repmat(uz, [nptot,1]);
tz   = (0:size(inz,1)-1)'/harm.fs;
outz = lsim(G0, inz, tz);
outz = outz + outnoise*randn(size(outz));

ms_zip = struct('harm',harm, 'nrofs',nrofsZ, ...
                'ex',sort([ex1(:);ex2(:)]), 'freq',fbin(:));
datB = iodata(outz, inz, 1/harm.fs, 'Period', nrofsZ, 'UserData', struct('ms',ms_zip));
datB = pretreat(datB, 'trans', trans);
Pb   = time2frf_ml(datB);                  % 2x2 frd (each column on its lines)

figure('Name','MIMO FRF: zippered single experiment');
G0b = frd(freqresp(G0, 2*pi*Pb.Frequency), Pb.Frequency, 'FrequencyUnit','Hz');
bode(G0b, Pb); grid on;
legend('true','estimated (zippered, 1 exp)','Location','best');
title('2\times2 FRF - zippered multisine, single experiment');

% NOTE: the orthogonal method gives the full 2x2 at every line. The zippered
% method uses a single record but each column is measured only on its own
% (every-other) lines; time2frf_ml interpolates each column onto the full grid
% so it plots as a continuous curve - the true per-channel resolution is half.

%% ===== STRUCTURED MODAL IDENTIFICATION (van der Hulst et al., MSSP 2026) =====
% mimobench IS a proportionally-damped, rank-one modal system, so frf2modal can
% recover the modal parameters directly from the orthogonal FRF estimate Pa
% (exactly as in Step_MIMO5).
[modal, Pm] = frf2modal(Pa, 0, numel(wn_true), 'initfreq', wn_true*1.05);

fprintf('\n--- identified modal parameters ---\n');
fprintf(' mode |  wn_true   wn_est [Hz] |  z_true    z_est\n');
for i = 1:numel(wn_true)
    fprintf('  %2d  | %8.2f  %8.2f    | %6.3f   %6.3f\n', ...
        i, wn_true(i), modal.wn(i), ze_true(i), modal.zeta(i));
end

% compare on the FRF grid so the axes match the data
G0m = frd(freqresp(G0, 2*pi*Pa.Frequency), Pa.Frequency, 'FrequencyUnit','Hz');
Pmf = frd(freqresp(Pm, 2*pi*Pa.Frequency), Pa.Frequency, 'FrequencyUnit','Hz');
figure('Name','Structured modal identification');
bode(G0m, Pa, Pmf); grid on;
legend('true','nonparametric FRF','identified modal model','Location','best');
title('2\times2 structured modal identification (frf2modal)');

% complex-FRF fit of the identified modal model vs the true plant
g0 = G0m.ResponseData(:);  gm = Pmf.ResponseData(:);
fprintf('modal model FRF fit vs true : %.2f %%\n', 100*(1-norm(gm-g0)/norm(g0-mean(g0))));

% --- general-viscous damping option (complex mode shapes, eqs.(2),(6),(46)) ---
% For non-proportionally damped systems use 'damping','general'; it also fits
% proportionally damped data (the complex mode shapes then come out ~real).
[modalG, PmG] = frf2modal(Pa, 0, numel(wn_true), ...
                          'damping','general', 'initfreq', wn_true*1.05);
PmGf = frd(freqresp(PmG, 2*pi*Pa.Frequency), Pa.Frequency, 'FrequencyUnit','Hz');
gG = PmGf.ResponseData(:);
fprintf('general-damping wn_est [Hz] :'); fprintf(' %.2f', modalG.wn); fprintf('\n');
fprintf('general modal FRF fit vs true: %.2f %%\n', 100*(1-norm(gG-g0)/norm(g0-mean(g0))));
