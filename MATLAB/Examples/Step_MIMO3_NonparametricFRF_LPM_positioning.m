%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MIMO NON-PARAMETRIC FRF (LOCAL POLYNOMIAL METHOD) - POSITIONING STAGE:
% ---------------------------------------------------------------------
% Descr.:   Full 2x2 FRF of the MIMO benchmark (mimobench) with the Local
%           Polynomial Method, from SHORT, transient-corrupted records.
%
%           Starting from rest, the lightly-damped resonances ring -> a
%           start-up transient corrupts the first periods. The LPM models the
%           transient from the non-excited spectral lines, so it returns a
%           low-bias 2x2 FRF from a SHORT record WITHOUT discarding the
%           transient periods. The classical ML estimator must discard those
%           periods first (wasted measurement time), and is biased if it keeps
%           them.
%
%           ORTHOGONAL multiple-experiment design (n_exp = n_in): every excited
%           line is driven in each experiment, so the FRF is resolved at FULL
%           resolution (each line) - the sharp resonances are captured.
%           (The zippered single-record alternative trades half the per-channel
%           resolution; see the note at the end.)
% System:   2x2 rank-one modal benchmark (mimobench), force -> position-like.
% Author:   Wataru Ohnishi, The University of Tokyo, 2026
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc;
[G0, wn_true, ze_true] = mimobench();        % 2x2 truth (modes ~40/95/180 Hz)
nrofi = 2;  nrofo = 2;

%% STEP 1: ORTHOGONAL EXCITATION DESIGN (n_exp = n_in experiments)
harm.fs = 2000;  harm.df = 1;  harm.fl = 5;  harm.fh = 400;  harm.fr = 1.02;
options.itp = 'r'; options.ctp = 'c'; options.dtp = 'f'; options.gtp = 'l';
ms    = multisine(harm, repmat(tf(1),[1,nrofi]), options);
nrofs = ms.nrofs;                             % samples per period (= fs/df)
nexp  = nrofi;                                % orthogonal -> n_in experiments
Tper  = 1/harm.df;                            % period length [s]

%% STEP 2: EXPERIMENTS (each from rest -> start-up transient + steady response)
P   = 6;        % periods per experiment
nL  = 3;        % early (transient) periods used by the LPM
nM  = 2;        % settled (late) periods used by the classical ML
t   = (0:P*nrofs-1)'/harm.fs;
ufull = cell(1,nexp);  yfull = cell(1,nexp);
rng('default');
for e = 1:nexp
    Ue = squeeze(ms.x(:,e,:)).';              % one period, nrofs x n_in
    ue = repmat(Ue, P, 1);
    ye = lsim(G0, ue, t);                     % from rest -> ringing + steady
    ye = ye + 1e-3*rms(ye(:))*randn(size(ye));
    ufull{e} = ue;  yfull{e} = ye;
end
datsel = @(sel) iodata(cellfun(@(Y) Y(sel,:), yfull, 'uni',0), ...
                       cellfun(@(U) U(sel,:), ufull, 'uni',0), ...
                       1/harm.fs, 'Period', nrofs, 'UserData', struct('ms', ms));

figure('Name','MIMO positioning experiment (time domain, exp 1)');
ax1 = subplot(211); plot(t, ufull{1}); ylabel('force [-]'); grid on;
    title(sprintf('%d orthogonal experiments x %d periods (%.1f s each)  |  LPM: first %d  ML: last %d', ...
                  nexp, P, P*Tper, nL, nM));
ax2 = subplot(212); plot(t, yfull{1}); ylabel('position [-]'); xlabel('time [s]'); grid on;
    legend('start-up transient + steady response','Location','best');
shade_periods(ax1, Tper, P, nL, nM, true);
shade_periods(ax2, Tper, P, nL, nM, false);

%% STEP 3: FRF ESTIMATION
% (a) MIMO LPM on the FIRST nL periods (during the transient), full resolution
selL = 1:nL*nrofs;
Plpm = time2frf_lpm(datsel(selL), 'order', 2, 'halfwidth', 2);

% (b) ML on the same short, transient-corrupted record -> biased
Pbias = time2frf_ml(datsel(selL));

% (c) classical ML done properly: discard the transient, average the SETTLED
%     periods at the end of the record.
selM = (P-nM)*nrofs+1 : P*nrofs;
Psettled = time2frf_ml(datsel(selM));

G0g = frd(freqresp(G0, 2*pi*Plpm.Frequency), Plpm.Frequency, 'FrequencyUnit','Hz');
figure('Name','MIMO 2x2 FRF: LPM vs ML');
bode(G0g, Plpm, Psettled, Pbias); grid on;
legend('true', ...
       sprintf('LPM (first %d periods, transient)', nL), ...
       sprintf('ML (settled, last %d periods)', nM), ...
       'ML (first periods, transient kept)', 'Location','best');
title('2\times2 FRF: LPM (short, transient) vs ML (settled) vs ML (biased)');

%% STEP 4: FRF ERROR AND FIT (vs the true plant)
%   Fit% = 100*(1 - ||G_est - G0|| / ||G0 - mean(G0)||)   (over the full 2x2)
fprintf('\n--- 2x2 FRF fit vs true plant ---\n');
fprintf('LPM (first %d periods, transient)   : %5.1f %%\n', nL, fitMIMO(Plpm,    G0));
fprintf('ML  (settled, last %d periods)      : %5.1f %%\n', nM, fitMIMO(Psettled,G0));
fprintf('ML  (first periods, transient kept) : %5.1f %%\n',     fitMIMO(Pbias,   G0));

f   = Plpm.Frequency;  G0r = freqresp(G0, 2*pi*f);
figure('Name','MIMO FRF error vs true plant');
for o = 1:nrofo
    for i = 1:nrofi
        subplot(nrofo, nrofi, (o-1)*nrofi + i);
        eL = squeeze(Plpm.ResponseData(o,i,:))     - squeeze(G0r(o,i,:));
        eS = squeeze(Psettled.ResponseData(o,i,:)) - squeeze(G0r(o,i,:));
        eB = squeeze(Pbias.ResponseData(o,i,:))    - squeeze(G0r(o,i,:));
        semilogx(f, mag2db(abs(eL)), 'LineWidth',1.2); hold on; grid on;
        semilogx(f, mag2db(abs(eS)));
        semilogx(f, mag2db(abs(eB)));
        xlim([harm.fl harm.fh]); title(sprintf('|G_{%d%d} error|',o,i));
        ylabel('[dB]'); if o==nrofo, xlabel('Frequency [Hz]'); end
        if o==1 && i==1, legend('LPM','ML settled','ML biased','Location','best'); end
    end
end

%% STEP 5: DATA EFFICIENCY
fprintf('\n--- measurement length ---\n');
fprintf('LPM : first %d periods incl. transient = %.1f s per experiment\n', nL, nL*Tper);
fprintf('ML  : discard transient, then average %d settled periods\n', nM);
fprintf('LPM uses every period (no discarded transient) at full resolution.\n');

% NOTE: same periodic LPM as the SISO positioning example. With ORTHOGONAL
%       experiments the transient-removed spectrum is solved into the full 2x2
%       at every excited line, so sharp resonances are resolved. The ZIPPERED
%       single-record design (one experiment, interleaved lines) is also
%       supported by time2frf_lpm, but its per-channel resolution is 1/n_in, so
%       a resonance sharper than the per-channel line spacing is missed - use
%       the orthogonal design here for sharply-resonant stages.

%% ===== local functions =====
function p = fitMIMO(P, G0)
g0 = reshape(freqresp(G0, 2*pi*P.Frequency), [], 1);
ge = P.ResponseData(:);
p  = 100*(1 - norm(ge-g0)/norm(g0-mean(g0)));
end

function shade_periods(ax, Tper, P, nL, nM, withText)
hold(ax,'on');  yl = get(ax,'YLim');  yb = [yl(1) yl(1) yl(2) yl(2)];
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
