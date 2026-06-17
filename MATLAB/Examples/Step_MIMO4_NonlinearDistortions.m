%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MIMO NONLINEAR DISTORTIONS (ROBUST BEST LINEAR APPROXIMATION):
% -------------------------------------------------------------
% Descr.:   Quantify the nonlinear distortion level of a 2x2 system with the
%           robust method: M independent realizations of a random-phase
%           multisine, each measured over several periods. The Best Linear
%           Approximation (BLA) is the average FRF over realizations. The
%           scatter ACROSS realizations carries noise + nonlinear distortions;
%           the scatter ACROSS PERIODS carries the noise only - subtracting
%           the two separates the stochastic nonlinear distortion level from
%           the measurement noise (time2bla).
% System:   2x2 modal benchmark (mimobench) + a static cubic output
%           nonlinearity (Wiener-type), so the response contains genuine
%           nonlinear distortions.
% Author:   Wataru Ohnishi, The University of Tokyo, 2026
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc;
[G0, wn_true, ze_true] = mimobench();        % 2x2 linear truth
nrofi = 2;  nrofo = 2;
nlc = 0.05;     % cubic output-distortion strength (relative to rms); tunable

%% STEP 1: EXCITATION (orthogonal multisine; phases re-randomized per realization)
% NOTE: multisine() seeds its random phases deterministically (per input), so
% every call returns the SAME phases. For the robust BLA we need INDEPENDENT
% realizations, so we rotate the excited-line phases ourselves: a per-input
% phase (the SAME across the n_in orthogonal experiments) keeps the Hadamard
% orthogonality (left-multiplying U by a unitary diagonal) while making each
% realization an independent random-phase multisine.
harm.fs = 2000;  harm.df = 1;  harm.fl = 5;  harm.fh = 400;  harm.fr = 1.02;
options.itp = 'r';  options.ctp = 'c';  options.dtp = 'f';  options.gtp = 'l';
M     = 6;          % number of random realizations
nrofp = 3;          % periods per experiment (>=2, for the noise estimate)
trans = 1;          % transient period (removed by pretreat)
nptot = nrofp + trans;
ms    = multisine(harm, repmat(tf(1),[1,nrofi]), options);
nrofs = ms.nrofs;
exb   = ms.ex(:);                                  % excited bins (1-based)
t     = (0:nptot*nrofs-1)'/harm.fs;

%% STEP 2: EXPERIMENTS (each realization = n_in orthogonal experiments)
yc = {};  uc = {};
for m = 1:M
    rng(100+m);
    phi = 2*pi*rand(nrofi, numel(exb));            % per-input excited-line phase
    for e = 1:nrofi
        Ue = zeros(nrofs, nrofi);
        for i = 1:nrofi
            Xi = fft(squeeze(ms.x(i,e,:)));        % one-period spectrum
            rot = ones(nrofs,1);
            rot(exb)          = exp(1i*phi(i,:).');
            rot(nrofs-exb+2)  = conj(exp(1i*phi(i,:).'));   % conjugate symmetry
            Ue(:,i) = real(ifft(Xi.*rot));
        end
        u  = repmat(Ue, nptot, 1);
        yl = lsim(G0, u, t);                       % linear response
        sc = rms(yl(:));
        y  = yl + nlc*sc*(yl/sc).^3;               % static cubic nonlinearity
        y  = y + 1e-3*sc*randn(size(y));           % measurement noise
        yc{end+1} = y;  uc{end+1} = u;             %#ok<AGROW>
    end
end
dat = iodata(yc, uc, 1/harm.fs, 'Period', nrofs, 'UserData', struct('ms', ms));
dat = pretreat(dat, 'trans', trans);

%% STEP 3: ROBUST BLA + DISTORTION ANALYSIS
bla = time2bla(dat, M);

%% STEP 4: BLA vs TRUE LINEAR PLANT
G0g = frd(freqresp(G0, 2*pi*bla.freq), bla.freq, 'FrequencyUnit','Hz');
figure('Name','MIMO BLA vs true linear plant');
bode(G0g, bla.G); grid on;
legend('true linear G_0','BLA (average over realizations)','Location','best');
title('2\times2 Best Linear Approximation');

%% STEP 5: NOISE vs NONLINEAR DISTORTION LEVELS (per entry)
f = bla.freq;
figure('Name','MIMO nonlinear distortion analysis');
for o = 1:nrofo
    for i = 1:nrofi
        subplot(nrofo, nrofi, (o-1)*nrofi + i);
        semilogx(f, mag2db(abs(squeeze(bla.G.ResponseData(o,i,:)))), 'LineWidth',1.2); hold on; grid on;
        semilogx(f, mag2db(squeeze(bla.sG_total(o,i,:))));
        semilogx(f, mag2db(squeeze(bla.sG_nl(o,i,:))));
        semilogx(f, mag2db(squeeze(bla.sG_noise(o,i,:))));
        xlim([harm.fl harm.fh]); title(sprintf('G_{%d%d}',o,i)); ylabel('[dB]');
        if o==nrofo, xlabel('Frequency [Hz]'); end
        if o==1 && i==1
            legend('|BLA|','total std','nonlinear std','noise std','Location','best');
        end
    end
end

%% STEP 6: SUMMARY
nl2noise = 20*log10( mean(bla.sG_nl(:)) / mean(bla.sG_noise(:)) );
fprintf('\n--- robust BLA distortion analysis ---\n');
fprintf('realizations M = %d , periods/experiment = %d , inputs = %d\n', M, nrofp, nrofi);
fprintf('mean nonlinear-to-noise level : %+.1f dB\n', nl2noise);
fprintf('(nonlinear distortions dominate the noise where this is well above 0 dB)\n');

% NOTE: the BLA is the linear model "seen" by this random-phase excitation; it
% differs from G0 by the describing-function gain of the nonlinearity. The
% nonlinear std curve is the level a parametric model (Step_MIMO5) cannot fit -
% it bounds the achievable fit. Set nlc=0 to recover a purely linear system
% (nonlinear std then collapses into the noise floor).
