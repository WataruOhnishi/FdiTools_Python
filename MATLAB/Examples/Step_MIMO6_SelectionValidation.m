%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MIMO MODEL SELECTION & VALIDATION (STRUCTURED MODAL):
% ----------------------------------------------------
% Descr.:   Select the number of flexible modes and validate the structured
%           modal model (frf2modal, Step_MIMO5) against the measured 2x2 FRF.
%           Selection: sweep the mode count and watch the noise-normalized
%           residual cost drop to its floor (the knee = the right order).
%           Validation: compare the modeling error |G_meas - G_model| to the
%           FRF uncertainty sigma_G per entry - an adequate model has its error
%           at/below the uncertainty (the achievable floor).
% System:   2x2 rank-one modal benchmark (mimobench).
% Author:   Wataru Ohnishi, The University of Tokyo, 2026
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc;
[G0, wn_true, ze_true] = mimobench();
nrofi = 2;  nrofo = 2;  nflex_true = numel(wn_true);

%% STEP 1: MEASURE THE 2x2 FRF (orthogonal multisine, with uncertainty)
harm.fs = 2500;  harm.df = 1;  harm.fl = 1;  harm.fh = 500;  harm.fr = 1.02;
options.itp = 'r'; options.ctp = 'c'; options.dtp = 'f'; options.gtp = 'l';
ms    = multisine(harm, repmat(tf(1),[1,nrofi]), options);
nrofs = ms.nrofs;  nexp = nrofi;  nrofp = 3;  trans = 1;  nptot = nrofp + trans;
rng('default');  t = (0:nptot*nrofs-1)'/harm.fs;
uc = cell(1,nexp);  yc = cell(1,nexp);
for j = 1:nexp
    Uj  = squeeze(ms.x(:,j,:)).';
    in  = repmat(Uj, [nptot,1]);
    out = lsim(G0, in, t) + 1e-3*randn(nptot*nrofs, 2);
    uc{j} = in;  yc{j} = out;
end
dat  = iodata(yc, uc, 1/harm.fs, 'Period', nrofs, 'UserData', struct('ms',ms));
dat  = pretreat(dat, 'trans', trans);
Pest = time2frf_ml(dat);                 % 2x2 frd with UserData.sG

%% STEP 2: ORDER SELECTION (sweep the number of flexible modes)
orders = 1:nflex_true+2;
cost   = nan(size(orders));
for k = orders
    try
        [~, Pmk] = frf2modal(Pest, 0, k);          % auto CMIF initialisation
        cost(k)  = normcost(Pest, Pmk);
    catch
        cost(k)  = NaN;                            % failed fit -> skip
    end
end
figure('Name','Model order selection');
semilogy(orders, cost, 'o-', 'LineWidth',1.2); grid on; hold on;
yline(1,'k--');                                    % noise floor (cost ~ 1)
xline(nflex_true,'r:','true order');
xlabel('number of flexible modes'); ylabel('noise-normalized residual cost');
title('Order selection: cost drops to the noise floor at the right order');

%% STEP 3: IDENTIFY THE SELECTED MODEL AND COMPARE TO THE FRF
nflex = nflex_true;
[modal, Pm] = frf2modal(Pest, 0, nflex, 'initfreq', wn_true*1.05);

fprintf('\n--- identified modal parameters (selected order = %d) ---\n', nflex);
fprintf(' mode |  wn_true   wn_est [Hz] |  z_true    z_est\n');
for i = 1:nflex
    fprintf('  %2d  | %8.2f  %8.2f    | %6.3f   %6.3f\n', ...
        i, wn_true(i), modal.wn(i), ze_true(i), modal.zeta(i));
end

G0g = frd(freqresp(G0, 2*pi*Pest.Frequency), Pest.Frequency, 'FrequencyUnit','Hz');
Pmg = frd(freqresp(Pm, 2*pi*Pest.Frequency), Pest.Frequency, 'FrequencyUnit','Hz');
figure('Name','Selected modal model vs FRF');
bode(G0g, Pest, Pmg); grid on;
legend('true','nonparametric FRF','modal model','Location','best');
title(sprintf('2\\times2 modal model (%d modes)', nflex));

%% STEP 4: RESIDUAL VALIDATION (modeling error vs FRF uncertainty)
f  = Pest.Frequency;  Gm = freqresp(Pm, 2*pi*f);
sG = Pest.UserData.sG;
figure('Name','Residual validation: error vs uncertainty');
for o = 1:nrofo
    for i = 1:nrofi
        subplot(nrofo, nrofi, (o-1)*nrofi + i);
        err = squeeze(Pest.ResponseData(o,i,:) - Gm(o,i,:));
        semilogx(f, mag2db(abs(squeeze(Pest.ResponseData(o,i,:)))), 'LineWidth',1.2); hold on; grid on;
        semilogx(f, mag2db(abs(err)));
        semilogx(f, mag2db(squeeze(sG(o,i,:))));
        xlim([harm.fl harm.fh]); title(sprintf('G_{%d%d}',o,i)); ylabel('[dB]');
        if o==nrofo, xlabel('Frequency [Hz]'); end
        if o==1 && i==1, legend('|G meas|','|modeling error|','uncertainty \sigma_G','Location','northeast'); end
    end
end

%% STEP 5: SUMMARY
fitpct = 100*(1 - norm(Pmg.ResponseData(:)-G0g.ResponseData(:)) / ...
                  norm(G0g.ResponseData(:)-mean(G0g.ResponseData(:))));
rcost  = normcost(Pest, Pm);
fprintf('\n--- validation summary ---\n');
fprintf('selected modes              : %d\n', nflex);
fprintf('modal-model FRF fit vs true : %.2f %%\n', fitpct);
fprintf('noise-normalized cost       : %.2f  (~1 means error at the noise floor)\n', rcost);

% NOTE: the order-selection cost falls steeply until the true number of modes,
% then flattens near 1 (the residual is at the measurement-noise level): adding
% more modes does not help. With real data, replace the noise floor by the
% nonlinear distortion floor from Step_MIMO4 (time2bla) as the achievable bound.

%% ===== local function =====
function c = normcost(Pest, Pm)
f   = Pest.Frequency;
Gm  = freqresp(Pm, 2*pi*f);
res = Pest.ResponseData - Gm;
sG  = Pest.UserData.sG;
c   = mean( abs(res(:)).^2 ./ max(sG(:),eps).^2 );
end
