%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NON-PARAMETRIC FRF (LOCAL POLYNOMIAL METHOD) - POSITIONING STAGE:
% ----------------------------------------------------------------
% Descr.:   FRF estimation of the benchmark positioning stage (force ->
%           velocity, mdl.Pv) with the Local Polynomial Method (LPM).
%           Starting from rest, the lightly-damped resonances ring -> a
%           start-up transient corrupts the first periods. The LPM models the
%           transient from the non-excited spectral lines of the P-period
%           record, so it gives a low-bias FRF from a SHORT record WITHOUT
%           discarding the transient periods. The classical ML estimator must
%           discard those periods first (wasted measurement time / data).
% System:   high-precision positioning stage benchmark (LTI).
% Author:   Wataru Ohnishi, The University of Tokyo, 2026
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc;
load('private/20160829_ident');     % benchmark model
P0 = mdl.Pv(1,1);                   % force -> velocity  (use mdl.Pp(1,1) for position)

%% STEP 1: EXCITATION DESIGN (multisine)
harm.fs = 10000;        % sampling frequency [Hz]
harm.df = 1;            % resolution [Hz] -> period = 1 s
harm.fl = 1;            % lowest excited frequency [Hz]
harm.fh = 1000;         % highest excited frequency [Hz]
harm.fr = 1.02;
options.itp = 'r'; options.ctp = 'c'; options.dtp = 'f'; options.gtp = 'l';
ms    = multisine(harm, tf(1), options);    % flat multisine
nrofs = ms.nrofs;                           % samples per period (= fs/df)
Tper  = 1/harm.df;                          % period length [s]

%% STEP 2: ONE EXPERIMENT (from rest -> start-up transient + steady response)
P    = 6;               % number of periods in the single experiment
nL   = 3;               % periods used by LPM (the early, transient ones)
nM   = 2;               % settled periods used by the classical ML (the late ones)
ampl = 1;               % multisine amplitude
u1   = squeeze(ms.x(1,1,:));
u    = ampl*repmat(u1, P, 1);
t    = (0:P*nrofs-1)'/harm.fs;
y    = lsim(P0, u, t);                       % from rest -> ringing transient + steady
y    = y + 1e-4*rms(y)*randn(size(y));       % measurement noise (relative)
dat  = @(sel) iodata(y(sel), u(sel), 1/harm.fs, 'Period', nrofs, ...
                     'UserData', struct('ms', ms));

figure('Name','Positioning experiment (time domain)');
ax1 = subplot(211); plot(t, u); ylabel('force [-]'); grid on;
    title(sprintf('Single experiment: %d periods = %.1f s  (LPM: first %d  |  ML: last %d)', ...
                  P, P*Tper, nL, nM));
ax2 = subplot(212); plot(t, y); ylabel('velocity [-]'); xlabel('time [s]'); grid on;
    legend('start-up transient + steady excitation response','Location','best');
shade_periods(ax1, Tper, P, nL, nM, true);
shade_periods(ax2, Tper, P, nL, nM, false);

%% STEP 3: FRF ESTIMATION
% (a) LPM on the FIRST nL periods (during the transient): the PN-point DFT
%     lets the LPM model the transient from the non-excited lines.
selL = 1:nL*nrofs;
Pest_lpm = time2frf_lpm(dat(selL), 'order', 2, 'halfwidth', 2);

% (b) ML on the same short, transient-corrupted record -> biased
Pest_ml_bias = time2frf_ml(dat(selL));

% (c) classical ML done properly: discard the transient, average the SETTLED
%     periods at the end of the record.
selM = (P-nM)*nrofs+1 : P*nrofs;
Pest_ml_settled = time2frf_ml(dat(selM));

bode_fdi({P0, Pest_lpm, Pest_ml_settled, Pest_ml_bias}, ...
    'legend', {'true', ...
               sprintf('LPM (first %d periods, transient)', nL), ...
               sprintf('ML (settled, last %d periods)', nM), ...
               'ML (first periods, transient kept)'}, ...
    'xlim',  [harm.fl harm.fh], 'unit', 'Hz', 'legendloc', 'southwest', ...
    'title', 'Stage FRF: LPM (short, transient) vs ML (settled) vs ML (biased)');

%% STEP 4: FRF ERROR AND FIT (vs the true plant)
%   Fit% = 100*(1 - ||G_est - G0|| / ||G0 - mean(G0)||)
fL = Pest_lpm.Frequency;        gL = squeeze(Pest_lpm.ResponseData);
fS = Pest_ml_settled.Frequency; gS = squeeze(Pest_ml_settled.ResponseData);
fB = Pest_ml_bias.Frequency;    gB = squeeze(Pest_ml_bias.ResponseData);
GtL = squeeze(freqresp(P0, 2*pi*fL));
GtS = squeeze(freqresp(P0, 2*pi*fS));
GtB = squeeze(freqresp(P0, 2*pi*fB));
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

%% STEP 5: DATA EFFICIENCY
fprintf('\n--- measurement length ---\n');
fprintf('LPM : first %d periods incl. transient   = %.1f s\n', nL, nL*Tper);
fprintf('ML  : discard transient then average %d   = at least %d periods\n', nM, nL);
fprintf('LPM uses every period (no discarded transient).\n');

% NOTE: same LPM as the thermal example (Step_3_..._thermal) - the periodic
%       PN-point DFT models the transient from the non-excited lines. Here the
%       transient is the lightly-damped stage ringing instead of a slow thermal
%       drift. Use mdl.Pp(1,1) to identify force->position instead of velocity.

%% ===== local function: shade the periods (LPM region / ML region) =====
function shade_periods(ax, Tper, P, nL, nM, withText)
hold(ax,'on');
yl = get(ax,'YLim');
yb = [yl(1) yl(1) yl(2) yl(2)];
patch('Parent',ax,'XData',[0 nL nL 0]*Tper,'YData',yb, ...
      'FaceColor',[0.1 0.4 1.0],'FaceAlpha',0.10,'EdgeColor','none','HandleVisibility','off');
patch('Parent',ax,'XData',([P-nM P P P-nM])*Tper,'YData',yb, ...
      'FaceColor',[1.0 0.4 0.1],'FaceAlpha',0.10,'EdgeColor','none','HandleVisibility','off');
for k = 1:P-1
    plot(ax,[k k]*Tper, yl, ':', 'Color',[0.6 0.6 0.6],'HandleVisibility','off');
end
set(ax,'YLim',yl);
uistack(findobj(ax,'Type','line','-not','LineStyle',':'),'top');
if withText
    text(ax, 0.5*nL*Tper,     yl(2), '  LPM (transient)', ...
         'VerticalAlignment','top','Color',[0.05 0.2 0.8],'FontWeight','bold');
    text(ax, (P-0.5*nM)*Tper, yl(2), 'ML (settled)  ', ...
         'VerticalAlignment','top','HorizontalAlignment','right', ...
         'Color',[0.8 0.3 0.0],'FontWeight','bold');
end
end
