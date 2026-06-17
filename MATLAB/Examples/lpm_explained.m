function lpm_explained
%LPM_EXPLAINED  Data-driven illustration of the Local Polynomial Method, using
% the Step_3 thermal benchmark. Shows (time) the huge start-up transient over
% the tiny ripple, and (frequency) how the transient is modelled from the
% NON-excited DFT bins by a local polynomial and subtracted at each excited
% line:  Y(K+m) = Y0*[m==0] + sum_s t_s m^s ,  Ghat = Y0 / U(K).
%
% LPM takes the chosen record (here the FIRST nL periods, as Step_3 does) and
% transforms it with ONE PN-point DFT over the whole block — it does NOT chop
% period-by-period (that is the ML/averaging route).
%
% Self-contained (needs only the Control System Toolbox). The figure is saved
% to docs/img/lpm_explained.png.
%   Author : Wataru Ohnishi, The University of Tokyo, 2026

%% --- plant & excitation (same numbers as Step_3 thermal) ----------------
s    = tf('s');
G0   = 50/((3600*s+1)*(120*s+1));        % heater [%] -> temperature [degC]
fs   = 0.1;   df = 5e-4;                  % Ts = 10 s, period = 1/df = 2000 s
fl   = 1e-3;  fh = 1e-2;                  % excited band [Hz]
nrofs= round(fs/df);                      % 200 samples / period
Tper = 1/df;                              % 2000 s/period
P    = 8;     nL = 4;   nM = 3;           % total periods; LPM first nL; ML last nM
DC   = 3;     ampl = 2;                   % heater bias / multisine amp [%]
R    = 2;     n  = 2;                     % transient order / half window (n<nL)
rng(2);

f1  = (0:nrofs-1)'*df;
exb = find(f1>=fl & f1<=fh & f1<fs/2);    % excited 1-period bins (1-based)
U1  = zeros(nrofs,1);  ph = exp(1i*2*pi*rand(numel(exb),1));
U1(exb) = ph;  U1(nrofs-exb+2) = conj(ph);
u1  = real(ifft(U1));  u1 = ampl*u1/max(abs(u1));

t = (0:P*nrofs-1)'/fs;
u = DC + repmat(u1,P,1);
y = lsim(G0,u,t);
y = y + 0.01*randn(size(y));

%% --- LPM record = first nL periods, ONE PN-point DFT over the block ------
PN   = nL*nrofs;
Y    = fft(y(1:PN));   U = fft(u(1:PN));
fbin = (0:PN-1)'*(fs/PN);
Kall = nL*(exb-1)+1;                       % excited lines: every nL-th PN bin
m    = (-n:n)';
Kr   = [double(m==0), m.^(0:R)];           % [spike , transient polynomial]

% pick the excited line with the LARGEST signal Y0 (clearest subtraction)
Y0all = zeros(numel(Kall),1);
for q = 1:numel(Kall), thq = Kr\Y(Kall(q)+m); Y0all(q) = thq(1); end
[~,qmax] = max(abs(Y0all));  K = Kall(qmax);
th = Kr \ Y(K+m);  Y0 = th(1);  t0 = th(2);
mm = linspace(-n,n,60)';  tcurve = (mm.^(0:R))*th(2:end);

%% ============================ FIGURE ====================================
figure('Color','w','Position',[60 60 1150 780]);

% (1) time domain with Step_3-style period shading
subplot(2,2,1);
plot(t/3600, y, 'b'); grid on; hold on;
yl = [0 165];
patch([0 nL nL 0]*Tper/3600,       [yl(1) yl(1) yl(2) yl(2)], [.1 .4 1], 'FaceAlpha',.10,'EdgeColor','none');
patch(([P-nM P P P-nM])*Tper/3600, [yl(1) yl(1) yl(2) yl(2)], [1 .4 .1], 'FaceAlpha',.10,'EdgeColor','none');
for k=1:P-1, xline(k*Tper/3600,':','Color',[.7 .7 .7]); end
ylim(yl);
text(nL*Tper/3600/2,     158,'LPM (first 4)','Color',[0 0 .7],'HorizontalAlignment','center','FontWeight','bold');
text((P-nM/2)*Tper/3600, 158,'ML (last 3)','Color',[.8 .3 0],'HorizontalAlignment','center','FontWeight','bold');
xlabel('time [h]'); ylabel('temperature [°C]');
title('Time: transient 0→150°C  ≫  ripple');

% (1b) input
subplot(2,2,3);
plot(t/3600, u, 'Color',[0 .55 0]); grid on;
xlabel('time [h]'); ylabel('heater [%]');
title('Input: DC bias + small multisine');

% (2) frequency: PN spectrum of the nL-period block
subplot(2,2,2);
mag = 20*log10(abs(Y)/PN + eps);
plot(fbin, mag, '-', 'Color',[.75 .75 .75]); hold on; grid on;
plot(fbin(Kall), mag(Kall), 'r.', 'MarkerSize',13);
plot(fbin(K), mag(K), 'ko', 'MarkerSize',9, 'LineWidth',1.3);
set(gca,'XScale','log'); xlim([fl/2 fh*1.4]);
xlabel('frequency [Hz]'); ylabel('|Y|/PN [dB]');
title('Freq: PN-DFT of the 4-period block (○ = zoomed line K)');

% (3) complex-plane view at line K:  Y(K) = t0 (transient) + Y0 (signal)
subplot(2,2,4); sc = 1/PN;
tc  = tcurve*sc;                         % fitted transient locus (complex)
mn  = m(m~=0);  Ynn = Y(K+mn)*sc;        % non-excited neighbour bins
t0c = t0*sc;    YKc = Y(K)*sc;
plot(real(tc),  imag(tc),  'm-', 'LineWidth',1.8); hold on; grid on;
plot(real(Ynn), imag(Ynn), 'o','Color',[.45 .45 .45],'MarkerFaceColor',[.45 .45 .45],'MarkerSize',7);
plot(real(t0c), imag(t0c), 'mo','MarkerSize',10,'LineWidth',1.8);
plot(real(YKc), imag(YKc), 'rs','MarkerFaceColor','r','MarkerSize',9);
quiver(real(t0c),imag(t0c), real(YKc-t0c),imag(YKc-t0c), 0, ...
       'g','LineWidth',2.5,'MaxHeadSize',0.7);
for q = 1:numel(mn)
    text(real(Ynn(q)),imag(Ynn(q)), sprintf('  K%+d',mn(q)), 'Color',[.4 .4 .4],'FontSize',9);
end
text(real(t0c),imag(t0c),'  t_0 (transient at K)','Color','m','VerticalAlignment','top');
text(real(YKc),imag(YKc),'  Y(K) = signal + transient','Color','r','VerticalAlignment','bottom');
text((real(t0c)+real(YKc))/2,(imag(t0c)+imag(YKc))/2, ...
     '  Y_0 = Y(K) - t_0  (signal)','Color',[0 .5 0],'FontWeight','bold');
% independent (non-equal) axis zoom so the cluster and the Y0 arrow fill the panel
allRe = real([Ynn; YKc; t0c; tc(:)]);  allIm = imag([Ynn; YKc; t0c; tc(:)]);
px = max(max(allRe)-min(allRe), eps);   py = max(max(allIm)-min(allIm), eps);
xlim([min(allRe)-0.20*px, max(allRe)+1.7*px]);   % extra right room for labels
ylim([min(allIm)-0.12*py, max(allIm)+0.12*py]);
xlabel('Re\{Y\}/PN'); ylabel('Im\{Y\}/PN');
title('Complex plane @ line K:  Y(K) = t_0 + Y_0   (bins K\pmm)');

%% --- printout -----------------------------------------------------------
Ghat = Y0/U(K);  Gtru = freqresp(G0, 2*pi*fbin(K));
fprintf('LPM record = first %d of %d periods; PN-point DFT (PN=%d) over the block.\n',nL,P,PN);
fprintf('zoom line K=%d  f=%.3f mHz : |t0|=%.3g, |Y0|=%.3g  -> |Ghat|=%.4g vs |G0|=%.4g\n', ...
        K, fbin(K)*1e3, abs(t0)/PN, abs(Y0)/PN, abs(Ghat), abs(Gtru));

%% --- save into the public docs image folder -----------------------------
try
    outpng = fullfile(fileparts(fileparts(mfilename('fullpath'))),'docs','img','lpm_explained.png');
    exportgraphics(gcf, outpng, 'Resolution',150);
    fprintf('saved %s\n', outpng);
catch ME
    fprintf(2,'save skipped: %s\n', ME.message);
end
end
