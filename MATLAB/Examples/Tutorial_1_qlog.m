%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TUTORIAL OF FDI by QLOG:
% ----------------------
% Descr.:   Tutorial of FdiTools by qlog excitation.
% System:   High-precision positioning stages with two encoders.
% Author:   Wataru Ohnishi, The University of Tokyo
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc;

%% STEP 1: ExcitationDesign
harm.fs = 10000;         % sampling frequency
harm.df = 1;            % frequency resolution
harm.fl = 1;            % lowest frequency
harm.fh = 1000;          % highest frequency
harm.fr = 1.02;         % frequency log ratio
% Design Options:
options.itp = 'r';      % init phase type:  s=schroeder/r=random
options.ctp = 'c';      % compression type: c=comp/n=no_comp
options.dtp = 'f';      % signal type:      f=full/ O=odd-odd
                        %                   o=odd / O2=special odd-odd
options.gtp = 'q';      % grid type: l=linear/q=quasi-logarithmic
% Ampliude spectrum:
nrofi = 1;              % Define number of inputs
Hampl = repmat(tf(1),[1,nrofi]); % flat spectrum
ms = multisine(harm, Hampl, options);

hfig=figure; sub = 0;
for ii = 1:nrofi
    for jj = 1:nrofi
        sub = sub+1;
        subplot(nrofi, nrofi, sub);
        plot(ms.time,squeeze(ms.x(ii,jj,:)))
        title(strcat('CF =',num2str(ms.cf(ii,jj))))
        old = axis; axis([0, ms.time(end), old(3:4)])
    end
end
hfig=figure; sub=0;
for ii = 1:nrofi
    for jj = 1:nrofi
        sub = sub+1;
        subplot(nrofi, nrofi, sub);
        semilogx(ms.freq,dbm(squeeze(ms.X(ii,jj,:))),'+')
        old = axis; axis([0, length(ms.x), old(3:4)])
    end
end

% header file output
% path = multisine2hdr(ms,'data/multisine.h');

%% EXPERIMENT
load('private/20160829_ident'); % load benchmark model
P0 = mdl.Pv(1,1);
nrofs = length(ms.x(1,1,:));
input = squeeze(ms.x(1,1,:));
nrofp = 5; % number of period of periodic excitation
input = repmat(input,[nrofp,1]);
input_noise_amp = 0.01; % amp of input noise 
input_noize = input + input_noise_amp*randn(size(input));
Ts = 1/ms.harm.fs;
t = 0:Ts:Ts*(length(input)-1);
output = lsim(P0,input_noize,t);
output_noise_amp = 0.001; % amp of output noise 
output_noize = output + output_noise_amp*randn(size(output));

%% STEP 2: NonparametricFRF
% remove transient periods, offsets and trends
trans = 1;                      % number of transient periods
trend = 0;                      % period trend removal flag
r0 = (1:ms.nrofs);                 % time visualization range
rn = (ms.nrofs*2+1:(2+1)*ms.nrofs);   % data visualization range
dat = iodata(output_noize, input, 1/ms.harm.fs, 'Period', ms.nrofs, 'UserData', struct('ms', ms));
dat = pretreat(dat, 'trans', trans, 'trend', trend);
x = dat.InputData; y = dat.OutputData;
time = (0:size(x,1)-1)'*dat.Ts;

figure
subplot(211), plot(time(r0),x(rn,:))
    title('input data'), legend('input force','Location','best'), ylim([-2,2])
subplot(212), plot(time(r0),y(rn,:))
    title('output data'), legend('stage position','Location','best')
    xlabel('time [s]')

Pest = time2frf_ml(dat);
bode_fdi({P0,Pest(1,1)},[Pest.freq,Pest.UserData.FRFn(:,1)], ...
    'legend',{'true','estimated frd','noise'});

% 95% circular confidence band on the measured FRF (Pintelon-Schoukens
% eq.2-40): radius = sigma_Ghat * frfconf(p, M). With the simulated noise and
% the few averaged periods here, the band is clearly visible.
p  = 0.95;
cf = frfconf(p, Pest.UserData.nrofp);
bode_fdi({P0, Pest(1,1)}, [Pest.freq, cf*Pest.UserData.sG(:,1)], 'style','band', ...
    'legend', {'true','FRF','95% bound'}, ...
    'title',  sprintf('FRF with %d%% confidence band (M = %d periods)', round(100*p), Pest.UserData.nrofp));

%% STEP 4: PARAMETRIC ESTIMATION
% deterministric/stochastic estimation with non-parametric noise model
n=7;                        % model order of denominator polynomial
mh=[4;]; ml=[0;];     % model orders of numerator polynomial
relvar=1e-10;               % relative variation of costfunction (stop)
iter=5e2;                   % maximum number of iterations (stop)
GN = 0;                     % Levenberg-Marquardt optimization
cORd = 'c';                 % continuous model identifaction
FRF_W = ones(size(squeeze(Pest.resp)));   % least squares weighting function
relax = 1;                  % relaxation factor for btls estimation
freq = Pest.freq;

% Deterministic methods: non-linear least squares
[SYS.nls,SYS.wls] = nlsfdi(Pest,FRF_W,n,mh,ml,iter,relvar,GN,cORd);
    FRF.nls = hfrf(SYS.nls,freq);
    FRF.wls = hfrf(SYS.wls,freq);
% Stochastic methods: maximum likelihood estimation
[SYS.ml,SYS.ls]  = mlfdi(Pest,n,mh,ml,iter,relvar,GN,cORd);
    FRF.ml = hfrf(SYS.ml,freq);
    FRF.ls = hfrf(SYS.ls,freq);

% Stochastic methods: bootstrapped total least squares
[SYS.btls,SYS.gtls] = btlsfdi(Pest,n,mh,ml,relax,iter,relvar,cORd);
    FRF.btls = hfrf(SYS.btls,freq);
    FRF.gtls = hfrf(SYS.gtls,freq);

bode_fdi({P0,Pest(1,1),SYS.wls,SYS.nls,SYS.ls},[Pest.freq,Pest.UserData.FRFn(:,1)], ...
    'legend',{'TRUE','FRF','WLS','NLS','LS','FRFn'});

bode_fdi({P0,Pest(1,1),SYS.ml,SYS.btls,SYS.gtls},[Pest.freq,Pest.UserData.sG(:,1)], ...
    'legend',{'TRUE','FRF','MLE','BTLS','GTLS','sG'});

% best estimator
bode_fdi({P0,Pest(1,1),SYS.btls},[Pest.freq,Pest.UserData.sG(:,1)], ...
    'legend',{'TRUE','FRF','BTLS','sG'});
