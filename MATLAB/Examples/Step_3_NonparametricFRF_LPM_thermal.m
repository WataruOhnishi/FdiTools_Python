%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NON-PARAMETRIC FRF (LOCAL POLYNOMIAL METHOD) - THERMAL PLANT:
% ------------------------------------------------------------
% Descr.:   FRF estimation of a slow thermal plant (semiconductor vertical
%           furnace, heater power -> temperature) with the Local Polynomial
%           Method (LPM), following Pintelon-Schoukens / Ohnishi et al.
%           The plant has a very large time constant, so the experiment is
%           dominated by a LONG start-up transient. The LPM models the
%           transient from the non-excited spectral lines of the P-period
%           record and identifies the FRF from a SHORT record taken DURING
%           the transient. The classical ML estimator instead needs the
%           furnace to SETTLE first (~5 time constants) before averaging, so
%           it needs much more measurement time.
%           The plant is LTI (no temperature dependency).
% Author:   Wataru Ohnishi, The University of Tokyo, 2026
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc;

%% PLANT (LTI): heater power [%] -> temperature [degC]
s    = tf('s');
tau1 = 3600;        % dominant (slow) time constant [s] -> long transient
tau2 = 120;         % faster thermal time constant [s]
Kdc  = 50;          % DC gain [degC/%]
G0   = Kdc/((tau1*s+1)*(tau2*s+1));

%% STEP 1: EXCITATION DESIGN (low-frequency multisine, mHz band)
% Only the DC level and the overall amplitude are tuned (no per-frequency
% shaping), as in the reference experiment.
harm.fs = 0.1;          % sampling frequency [Hz]  (Ts = 10 s)
harm.df = 5e-4;         % resolution [Hz] -> period = 1/df = 2000 s (~33 min)
harm.fl = 1e-3;         % lowest excited frequency [Hz]
harm.fh = 1e-2;         % highest excited frequency [Hz]
harm.fr = 1.02;
options.itp = 'r'; options.ctp = 'c'; options.dtp = 'f'; options.gtp = 'l';
ms    = multisine(harm, tf(1), options);    % flat multisine
nrofs = ms.nrofs;                           % samples per period (= fs/df)
Tper  = 1/harm.df;                          % period length [s]

%% STEP 2: ONE EXPERIMENT (cold start -> long transient + small ripple)
P    = 8;               % number of periods in the single experiment
nL   = 4;               % periods used by LPM (the early, transient ones)
nM   = 3;               % settled periods used by the classical ML (the late ones)
DC   = 3;               % heater bias [%] (heats the furnace up)
ampl = 2;               % multisine perturbation amplitude [%]
u1   = squeeze(ms.x(1,1,:));
u    = DC + ampl*repmat(u1, P, 1);
t    = (0:P*nrofs-1)'/harm.fs;
y    = lsim(G0, u, t);                      % from cold -> long transient + ripple
y    = y + 0.01*randn(size(y));            % temperature-sensor noise [degC]
dat  = @(sel) iodata(y(sel), u(sel), 1/harm.fs, 'Period', nrofs, ...
                     'UserData', struct('ms', ms));

figure('Name','Thermal experiment (time domain)');
ax1 = subplot(211); plot(t/3600, u); ylabel('heater [%]'); grid on;
    title(sprintf('Single experiment: %d periods = %.1f h  (LPM: first %d  |  ML: last %d)', ...
                  P, P*Tper/3600, nL, nM));
ax2 = subplot(212); plot(t/3600, y); ylabel('temp [\circC]'); xlabel('time [h]'); grid on;
    legend('long transient + small excitation response','Location','best');
shade_periods(ax1, Tper, P, nL, nM, true);     % period bands + LPM/ML regions (+labels)
shade_periods(ax2, Tper, P, nL, nM, false);

%% STEP 3: FRF ESTIMATION
% (a) LPM on the FIRST nL periods (taken DURING the transient): the PN-point
%     DFT lets the LPM model the transient from the non-excited lines.
selL = 1:nL*nrofs;
Pest_lpm = time2frf_lpm(dat(selL), 'order', 2, 'halfwidth', 2);

% (b) ML on the same short, transient-corrupted record -> biased
Pest_ml_bias = time2frf_ml(dat(selL));

% (c) classical ML done properly: discard the transient and average the
%     SETTLED periods at the end of the record (the "wait, then measure" way).
selM = (P-nM)*nrofs+1 : P*nrofs;
Pest_ml_settled = time2frf_ml(dat(selM));

bode_fdi({G0, Pest_lpm, Pest_ml_settled, Pest_ml_bias}, ...
    'legend', {'true', ...
               sprintf('LPM (first %d periods, transient)', nL), ...
               sprintf('ML (settled, last %d periods)', nM), ...
               'ML (first periods, transient kept)'}, ...
    'xlim',  [harm.fl harm.fh], 'unit', 'Hz', 'legendloc', 'southwest', ...
    'title', 'Furnace FRF: LPM (short, transient) vs ML (settled) vs ML (biased)');

%% STEP 4: FRF ERROR AND FIT (vs the true plant)
% error |G_est - G0| in the frequency domain, and an FRF fit-rate
%   Fit% = 100*(1 - ||G_est - G0|| / ||G0 - mean(G0)||)
fL = Pest_lpm.Frequency;        gL = squeeze(Pest_lpm.ResponseData);
fS = Pest_ml_settled.Frequency; gS = squeeze(Pest_ml_settled.ResponseData);
fB = Pest_ml_bias.Frequency;    gB = squeeze(Pest_ml_bias.ResponseData);
GtL = squeeze(freqresp(G0, 2*pi*fL));
GtS = squeeze(freqresp(G0, 2*pi*fS));
GtB = squeeze(freqresp(G0, 2*pi*fB));
fitpct = @(g,Gt) 100*(1 - norm(g-Gt)/norm(Gt-mean(Gt)));
fprintf('\n--- FRF fit vs true plant ---\n');
fprintf('LPM (first %d periods, transient)   : %5.1f %%\n', nL, fitpct(gL,GtL));
fprintf('ML  (settled, last %d periods)      : %5.1f %%\n', nM, fitpct(gS,GtS));
fprintf('ML  (first periods, transient kept) : %5.1f %%\n', fitpct(gB,GtB));

figure('Name','FRF error vs true plant');
semilogx(fL, mag2db(abs(gL-GtL)), 'LineWidth',1.2); hold on; grid on;
semilogx(fS, mag2db(abs(gS-GtS)));
semilogx(fB, mag2db(abs(gB-GtB)));
xlabel('Frequency [Hz]'); ylabel('|G_{est} - G_0| [dB]');
legend(sprintf('LPM error (Fit %.1f%%)', fitpct(gL,GtL)), ...
       sprintf('ML settled error (Fit %.1f%%)', fitpct(gS,GtS)), ...
       sprintf('ML biased error (Fit %.1f%%)', fitpct(gB,GtB)), ...
       'Location','best');
title('FRF error w.r.t. the true plant');

%% STEP 5: EXPERIMENT-TIME SAVING
T_lpm = nL*Tper;                            % LPM: short record during transient
T_ml  = 5*tau1 + nM*Tper;                   % ML: settle ~5 tau, then measure
fprintf('\n--- experiment time ---\n');
fprintf('LPM (first %d periods, transient modelled): %6.0f s = %.1f h\n', nL, T_lpm, T_lpm/3600);
fprintf('ML  (wait ~5*tau to settle, then measure) : %6.0f s = %.1f h\n', T_ml, T_ml/3600);
fprintf('time saved by LPM                         : %.1f h (%.0f%%)\n', ...
    (T_ml-T_lpm)/3600, 100*(T_ml-T_lpm)/T_ml);

% NOTE: LPM uses the P-period DFT, where the bins between the excited lines
%       carry only the transient; fitting a local polynomial there removes the
%       transient and yields the FRF from the short, transient-corrupted
%       record. The ML estimate on the same record is biased, and a correct ML
%       needs the furnace to settle first - many extra hours. Tune ampl/noise
%       (SNR) and halfwidth/order if needed.

%% ===== local function: shade the periods (LPM region / ML region) =====
function shade_periods(ax, Tper, P, nL, nM, withText)
hold(ax,'on');
yl = get(ax,'YLim');  h = 3600;
yb = [yl(1) yl(1) yl(2) yl(2)];
% LPM region = first nL periods (early, transient)
patch('Parent',ax,'XData',[0 nL nL 0]*Tper/h,'YData',yb, ...
      'FaceColor',[0.1 0.4 1.0],'FaceAlpha',0.10,'EdgeColor','none','HandleVisibility','off');
% ML region = last nM periods (settled)
patch('Parent',ax,'XData',([P-nM P P P-nM])*Tper/h,'YData',yb, ...
      'FaceColor',[1.0 0.4 0.1],'FaceAlpha',0.10,'EdgeColor','none','HandleVisibility','off');
% period boundaries
for k = 1:P-1
    plot(ax,[k k]*Tper/h, yl, ':', 'Color',[0.6 0.6 0.6],'HandleVisibility','off');
end
set(ax,'YLim',yl);
uistack(findobj(ax,'Type','line','-not','LineStyle',':'),'top');   % data line on top
if withText
    text(ax, 0.5*nL*Tper/h,      yl(2), '  LPM (transient)', ...
         'VerticalAlignment','top','Color',[0.05 0.2 0.8],'FontWeight','bold');
    text(ax, (P-0.5*nM)*Tper/h,  yl(2), 'ML (settled)  ', ...
         'VerticalAlignment','top','HorizontalAlignment','right', ...
         'Color',[0.8 0.3 0.0],'FontWeight','bold');
end
end
