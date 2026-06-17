%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MIMO NON-PARAMETRIC FRF:
% ------------------------
% Descr.:   Full 2x2 FRF-matrix estimation of a MIMO mechanical system with
%           two excitation strategies:
%             - ORTHOGONAL multisine over multiple experiments (full rank U)
%             - ZIPPERED multisine in a single experiment (interleaved lines)
%           In both cases time2frf_ml solves G = Y/U for the complete transfer
%           matrix; the orthogonal per-entry 95% confidence band is shown.
% System:   2x2 rank-one modal benchmark (mimobench).
% Author:   Wataru Ohnishi, The University of Tokyo, 2026
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc;
G0 = mimobench();                       % 2x2 truth

%% STEP 1: MIMO EXCITATION (orthogonal multisine -> nrofi experiments)
harm.fs = 2500;  harm.df = 1;  harm.fl = 1;  harm.fh = 500;  harm.fr = 1.02;
options.itp = 'r'; options.ctp = 'c'; options.dtp = 'f'; options.gtp = 'l';
nrofi = 2;
ms    = multisine(harm, repmat(tf(1),[1,nrofi]), options);
nrofs = ms.nrofs;  nexp = nrofi;
nrofp = 10;  trans = 1;  nptot = nrofp + trans;

%% STEP 2: EXPERIMENTS (simulate each orthogonal experiment)
rng('default');
t  = (0:nptot*nrofs-1)'/harm.fs;
uc = cell(1,nexp);  yc = cell(1,nexp);
for j = 1:nexp
    Uj  = squeeze(ms.x(:,j,:)).';       % one period, nrofs x nrofi
    in  = repmat(Uj, [nptot,1]);
    out = lsim(G0, in, t);
    out = out + 1e-2*randn(size(out));
    uc{j} = in;  yc{j} = out;
end
% multi-experiment iodata; time2frf_ml auto-detects MIMO (nexp >= nrofi)
dat = iodata(yc, uc, 1/harm.fs, 'Period', nrofs, 'UserData', struct('ms',ms));
dat = pretreat(dat, 'trans', trans);

%% STEP 3: ORTHOGONAL MIMO FRF (multiple experiments)
Pest = time2frf_ml(dat);                % full 2x2 frd (orthogonal)

%% STEP 4: ZIPPERED MIMO FRF (single experiment, interleaved lines)
% input 1 -> odd excited lines, input 2 -> even lines: one record gives the 2x2
nrofsZ = round(harm.fs/harm.df);
fbin   = (0:nrofsZ/2-1)*harm.df;
inband = find(fbin >= harm.fl & fbin <= harm.fh);
exz = {inband(1:2:end), inband(2:2:end)};
U = zeros(nrofsZ, nrofi);  rng('default');
for i = 1:nrofi
    ph = exp(1i*2*pi*rand(numel(exz{i}),1));
    U(exz{i},i) = ph;  U(nrofsZ-exz{i}+2,i) = conj(ph);   % conj-symmetric -> real
end
uz   = real(ifft(U));  uz = uz./max(abs(uz),[],1);
inz  = repmat(uz, [nptot,1]);
tz   = (0:size(inz,1)-1)'/harm.fs;
outz = lsim(G0, inz, tz);  outz = outz + 1e-2*randn(size(outz));
ms_zip = struct('harm',harm, 'nrofs',nrofsZ, ...
                'ex',sort([exz{1}(:);exz{2}(:)]), 'freq',fbin(:));
datZ = iodata(outz, inz, 1/harm.fs, 'Period', nrofsZ, 'UserData', struct('ms',ms_zip));
datZ = pretreat(datZ, 'trans', trans);
Pz   = time2frf_ml(datZ);               % full 2x2 frd (zippered)

%% STEP 5: COMPARE BOTH AGAINST THE TRUE PLANT
G0g = frd(freqresp(G0, 2*pi*Pest.Frequency), Pest.Frequency, 'FrequencyUnit','Hz');
figure('Name','MIMO 2x2 FRF - orthogonal vs zippered');
bode(G0g, Pest, Pz); grid on;
legend('true','orthogonal (2 exp)','zippered (1 exp)','Location','best');
title('2\times2 FRF - orthogonal vs zippered');

g0o = reshape(freqresp(G0,2*pi*Pest.Frequency),[],1);
g0z = reshape(freqresp(G0,2*pi*Pz.Frequency),[],1);
fprintf('FRF fit  orthogonal: %.2f %%   zippered: %.2f %%\n', ...
    100*(1-norm(Pest.ResponseData(:)-g0o)/norm(g0o-mean(g0o))), ...
    100*(1-norm(Pz.ResponseData(:)-g0z)/norm(g0z-mean(g0z))));

%% STEP 6: FRF UNCERTAINTY (orthogonal, per-entry 95% confidence band)
% time2frf_ml stores the FRF standard deviation of every entry in UserData.sG
% (propagated from the period-to-period output noise). The 95% circular
% confidence radius is sG * frfconf(p, M)  (M periods per experiment).
sG = Pest.UserData.sG;  M = Pest.UserData.nrofp;  cf = frfconf(0.95, M);
fH = Pest.Frequency;
figure('Name','MIMO 2x2 FRF with 95% confidence band');
for o = 1:2
    for i = 1:2
        ax = subplot(2,2,(o-1)*2+i); set(ax,'XScale','log'); hold(ax,'on'); grid(ax,'on');
        g  = squeeze(Pest.ResponseData(o,i,:));
        sg = squeeze(sG(o,i,:));
        gdb = mag2db(abs(g));
        up = mag2db(abs(g)+cf*sg);  lo = mag2db(max(abs(g)-cf*sg, eps));
        fill(ax,[fH;flipud(fH)],[up;flipud(lo)],[0.6 0.6 0.6], ...
             'EdgeColor','none','FaceAlpha',0.3,'HandleVisibility','off');
        plot(ax, fH, gdb);
        xlim(ax,[harm.fl harm.fh]); ylabel(ax,'|G| [dB]');
        % auto y-limits from the FRF magnitude (so the band's lower edge, which
        % dives towards -inf where |G| is small, does not stretch the axis)
        ylim(ax, [floor((min(gdb)-15)/10)*10, ceil((max(gdb)+15)/10)*10]);
        title(ax, sprintf('G_{%d%d}',o,i));
        if o==2, xlabel(ax,'Frequency [Hz]'); end
    end
end

% NOTE: orthogonal uses nrofi experiments at full per-channel resolution;
% zippered uses a single record but each column is on its own (every-other)
% lines (interpolated for display). The estimated frd Pest feeds the structured
% modal identification in Step_MIMO5.
