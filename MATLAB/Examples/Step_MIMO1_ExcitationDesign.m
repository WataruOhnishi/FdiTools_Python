%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MIMO EXCITATION DESIGN:
% -----------------------
% Descr.:   Orthogonal multisine design for a 2-input MIMO experiment.
%           multisine(nrofi) generates nrofi orthogonal (Hadamard) experiments;
%           in each experiment all inputs are driven with phase patterns that
%           make the input matrix invertible at every excited frequency, which
%           time2frf_ml exploits to estimate the full MIMO FRF (Step_MIMO2).
% Author:   Wataru Ohnishi, The University of Tokyo, 2026
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc;

%% MIMO MULTISINE DESIGN
harm.fs = 2500;        % sampling frequency [Hz]
harm.df = 1;           % frequency resolution [Hz]
harm.fl = 1;           % lowest excited frequency [Hz]
harm.fh = 500;         % highest excited frequency [Hz]
harm.fr = 1.02;        % quasi-log ratio (unused for 'l')
options.itp = 'r'; options.ctp = 'c'; options.dtp = 'f'; options.gtp = 'l';
nrofi = 2;                              % number of inputs
Hampl = repmat(tf(1),[1,nrofi]);       % flat amplitude per input
ms    = multisine(harm, Hampl, options);
nexp  = nrofi;                          % orthogonal -> nrofi experiments
ex    = ms.ex;                          % excited line indices

% ms.x : (input x experiment x time) , ms.X : (input x experiment x freq)
fprintf('MIMO multisine: %d inputs, %d orthogonal experiments, %d excited lines\n', ...
        nrofi, nexp, numel(ex));

%% TIME-DOMAIN SIGNALS (one period, per input and experiment)
figure('Name','MIMO multisine - time domain');
for j = 1:nexp
    for i = 1:nrofi
        subplot(nrofi, nexp, (i-1)*nexp + j);
        plot(ms.time, squeeze(ms.x(i,j,:))); grid on;
        title(sprintf('u_%d, exp %d  (CF = %.2f)', i, j, ms.cf(i,j)));
        if i==nrofi, xlabel('time [s]'); end
    end
end

%% FREQUENCY-DOMAIN SIGNALS (excited lines)
figure('Name','MIMO multisine - spectra');
for j = 1:nexp
    for i = 1:nrofi
        subplot(nrofi, nexp, (i-1)*nexp + j);
        semilogx(ms.freq(ex), dbm(squeeze(ms.X(i,j,ex))), '.'); grid on;
        title(sprintf('|U_%d| exp %d', i, j)); xlim([harm.fl harm.fh]);
        ylabel('|U| [dB]'); if i==nrofi, xlabel('Frequency [Hz]'); end
    end
end

% NOTE: a single broadband experiment cannot identify a 2x2 FRF (the 2x2 input
% matrix would be rank 1 per frequency). The orthogonal design above spreads
% the excitation over nrofi experiments so the input matrix is full rank.

%% ZIPPERED MULTISINE (single experiment, interleaved lines per input)
% Alternative single-record design: each input excites a DISJOINT subset of the
% excited lines (input 1 -> odd, input 2 -> even), so one experiment yields the
% full 2x2. The per-channel frequency resolution is 1/nrofi of the orthogonal
% design. (Built directly via an inverse DFT; see also Tutorial_4_MIMO.)
nrofsZ = round(harm.fs/harm.df);
fbin   = (0:nrofsZ/2-1)*harm.df;                  % freq of bin k (1-based) = (k-1)*df
inband = find(fbin >= harm.fl & fbin <= harm.fh); % 1-based excited bins
exz = {inband(1:2:end), inband(2:2:end)};         % zipper the lines per input
U   = zeros(nrofsZ, nrofi);
rng('default');
for i = 1:nrofi
    ph = exp(1i*2*pi*rand(numel(exz{i}),1));
    U(exz{i},i)        = ph;                       % excited lines
    U(nrofsZ-exz{i}+2,i) = conj(ph);              % conjugate-symmetric -> real signal
end
uz = real(ifft(U));  uz = uz./max(abs(uz),[],1);  % normalize amplitude per input
tz = (0:nrofsZ-1)'/harm.fs;
Xz = fft(uz)/sqrt(nrofsZ);                         % spectra for plotting

figure('Name','Zippered multisine - single experiment');
for i = 1:nrofi
    subplot(nrofi,2,(i-1)*2+1);
    plot(tz, uz(:,i)); grid on; ylabel(sprintf('u_%d',i));
    if i==1, title('time domain (one period)'); end
    if i==nrofi, xlabel('time [s]'); end
    subplot(nrofi,2,(i-1)*2+2);
    semilogx(fbin(exz{i}), dbm(Xz(exz{i},i)), '.'); grid on;
    xlim([harm.fl harm.fh]); ylabel('|U| [dB]');
    if i==1, title('spectrum (own interleaved lines)'); end
    if i==nrofi, xlabel('Frequency [Hz]'); end
end

% NOTE: Step_MIMO2 estimates the FRF from the orthogonal experiments; the
% zippered design is the single-experiment alternative (Tutorial_4_MIMO).
