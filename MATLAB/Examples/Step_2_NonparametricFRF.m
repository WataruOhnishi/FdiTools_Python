%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NON-PARAMETRIC FRF:
% -------------------
% Descr.:   example of measurement data pre-processing
%           and non-parametric synchronized FRF estimation
% System:   Conventional motor-bench with flexible coupling
% Author:   Thomas Beauduin, KULeuven, PMA division, 2014
%           Wataru Ohnishi, The University of Tokyo, 2019
% Note  :   v2 uses the iddata-compatible iodata container.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc;
load('private/MultisineTypeA.mat');     % schoeder multisine experiment

%% STEP 1: TIME TREATMENT
% Build an iodata container (Ts, Period and multisine info are carried
% along with the data), then remove transient periods, offsets and trends.
trans = 1;                      % number of transient periods
trend = 0;                      % period trend removal flag
input  = [iq_adx];              % input data to motor bench
output = [theta_mx,-theta_my];  % output data of motor bench

dat = iodata(output, input, 1/ms.harm.fs, 'Period', ms.nrofs, 'UserData', struct('ms', ms));
dat.InputName  = {'iq'};
dat.OutputName = {'theta_m','theta_l'};
dat = pretreat(dat, 'trans', trans, 'trend', trend);

x = dat.InputData; y = dat.OutputData;
nrofs = dat.Period;             % samples per period
time  = (0:nrofs-1)'*dat.Ts;    % one-period time axis
nrofp = nperiods(dat);          % number of periods after transient removal

figure;
subplot(311)
for k = 1:nrofp
    h = plot(time, x(nrofs*(k-1)+1:k*nrofs)); hold on;
    h.DisplayName = sprintf('period%d',k+trans);
end
ylabel('input current [A]');
title(sprintf('%d periods (%d transient removal)',nrofp,trans));
% legend;
subplot(312)
for k = 1:nrofp
    h = plot(time, y(nrofs*(k-1)+1:k*nrofs,1)); hold on;
    h.DisplayName = sprintf('period%d',k+trans);
end
ylabel('motor angle [rad]');
subplot(313)
for k = 1:nrofp
    h = plot(time, y(nrofs*(k-1)+1:k*nrofs,2)); hold on;
    h.DisplayName = sprintf('period%d',k+trans);
end
ylabel('load angle [rad]');
xlabel('time [s]');

pause

%% STEP 2: NON-PARAMETRIC ESTIMATION
% fft data and vizualize in freq domain position data
flagTime = true;
Pest = time2frf_ml(dat, flagTime);

% FRF with its standard deviation sigma_Ghat (Pintelon-Schoukens eq.2-38,
% stored in UserData.sG).
bode_fdi(Pest(1,1), [Pest.freq, Pest.UserData.sG(:,1)], ...
    'legend', {'FRF','\sigma_{Ghat}'}, 'title', 'Motor-side');
bode_fdi(Pest(2,1), [Pest.freq, Pest.UserData.sG(:,2)], ...
    'legend', {'FRF','\sigma_{Ghat}'}, 'title', 'Load-side');

% 95% circular confidence band (eq.2-40): radius = sigma_Ghat * frfconf(p,M)
% With this high-SNR record (many averaged periods M) the band is tight; it
% widens at the anti-resonance, where the response - and thus the SNR - drops.
p  = 0.95;
cf = frfconf(p, Pest.UserData.nrofp);
bode_fdi(Pest(1,1), [Pest.freq, cf*Pest.UserData.sG(:,1)], 'style','band', ...
    'legend', {'FRF','95% bound'}, 'title', 'Motor-side with 95% confidence band');

% Fewer averaged periods -> larger uncertainty (sigma_Ghat ~ 1/sqrt(M)), so the
% band becomes clearly visible. Reuse the first few pre-treated periods.
nuse = min(3, floor(size(x,1)/nrofs));
sel  = 1:(nuse*nrofs);
dat_few  = iodata(y(sel,:), x(sel), 1/ms.harm.fs, 'Period', nrofs, ...
                  'UserData', struct('ms', ms));
Pest_few = time2frf_ml(dat_few);
cf2 = frfconf(p, Pest_few.UserData.nrofp);
bode_fdi(Pest_few(1,1), [Pest_few.freq, cf2*Pest_few.UserData.sG(:,1)], 'style','band', ...
    'legend', {sprintf('FRF (%d periods)',nuse),'95% bound'}, ...
    'title',  sprintf('Motor-side, %d periods: wider 95%% band', nuse));
