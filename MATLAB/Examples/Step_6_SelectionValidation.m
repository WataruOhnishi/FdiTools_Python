%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PARAMETRIC ESTIMATION:
% ----------------------
% Descr.:   Example of parametric system model estimation
%           from frequency-domain data with known noise model.
% System:   Conventional motor-bench with flexible coupling.
% Author:   Thomas Beauduin, KULeuven, PMA division, 2014
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc;
load('private/MultisineTypeA.mat');     % schoeder multisine experiment

% Time treatment: build an iodata container, then remove
% transients/offsets/trends (TypeA carries the multisine ms).
trans = 1;                      % number of transient periods
trend = 0;                      % period trend removal flag
input  = [iq_adx];              % input data to motor bench
output = [theta_mx,-theta_my];  % output data of motor bench

dat = iodata(output, input, 1/ms.harm.fs, 'Period', ms.nrofs, 'UserData', struct('ms', ms));
dat.InputName  = {'iq'};
dat.OutputName = {'theta_m','theta_l'};
dat = pretreat(dat, 'trans', trans, 'trend', trend);

% Non-parametric estimation of frf matrix data (keep time data for residuals)
Pest = time2frf_ml(dat, true);

% deterministric/stochastic estimation with non-parametric noise model
n=4;                        % model order of denominator polynomial
mh=[2;0]; ml=[0;0];         % model orders of numerator polynomial
relvar=1e-10;               % relative variation of costfunction (stop)
iter=5e2;                   % maximum number of iterations (stop)
GN = 0;                     % Levenberg-Marquardt optimization
cORd = 'c';                 % continuous model identifaction
FRF_W = ones(size(squeeze(Pest.resp)));   % least squares weighting function
relax = 1;                  % relaxation factor for btls estimation

% Deterministic method: non-linear least squares
[SYS.nls,SYS.wls] = nlsfdi(Pest,FRF_W,n,mh,ml,iter,relvar,GN,cORd);

% Stochastic method: maximum likelihood estimation
[SYS.ml,SYS.ls] = mlfdi(Pest,n,mh,ml,iter,relvar,GN,cORd);

% Stochastic method: bootstrapped total least squares
[SYS.btls,SYS.gtls] = btlsfdi(Pest,n,mh,ml,relax,iter,relvar,cORd);

% PART 5: SELECTION & VALIDATION
% Extract the raw arrays the validation tests need from the iodata/Pest
freq = Pest.freq;
fsHz = 1/dat.Ts;
x = dat.InputData;  y = dat.OutputData;
X   = Pest.UserData.X;   Y   = Pest.UserData.Y;
sX2 = Pest.UserData.sX2; sY2 = Pest.UserData.sY2; cXY = Pest.UserData.cXY;
sCR = Pest.UserData.sCR;
nrofo = size(Y,2);
FRFs = zeros(numel(freq),nrofo);
for o = 1:nrofo, FRFs(:,o) = squeeze(Pest.resp(o,1,:)); end
nrofp = nperiods(dat);          % number of measured periods

%% TEST 1: RESIDUALS
% whiteness test with fraction above the x%-percentile confidence bound
[lags,corr,cb50,frac50,tag,cb95,frac95] = residtest(x,y,freq,FRFs,SYS,sCR,fsHz);
figure 
plot(lags,corr(:,:,1),'.',lags,cb95,'k',lags,cb50,'k--')
    title(strcat(tag(:,1),' -- p_{>95%}:',num2str(frac95(:,1)),...
          ' _ - _ p_{>50%}:',num2str(frac50(:,1))));
    legend([tag(:,1);'frac_{95%}';'frac_{50%}'],'Location','best');
    xlabel('Lag number'), ylabel('Amplitude [dB]'), ylim([0,1.5])

figure
plot(lags,corr(:,:,2),'.',lags,cb95,'k',lags,cb50,'k--')
    title(strcat(tag(:,1),' -- p_{>95%}:',num2str(frac95(:,2)),...
          ' _ - _ p_{>50%}:',num2str(frac50(:,2))));
    legend([tag(:,1);'frac_{95%}';'frac_{50%}'],'Location','best');
    xlabel('Lag number'), ylabel('Amplitude [dB]'),ylim([0,1.5])
pause

%% TEST 2: COST FUNCTION
% residual estimation cost (maximum likelihood function)
[cost,intv,tag] = costtest(X,Y,freq,sX2,sY2,cXY,SYS,relax,nrofp);
figure
bar(cost,0.8), hold on
line([0,length(cost)+1],[intv(2),intv(2)],'Color','k','LineStyle','--')
    set(gca,'xticklabel',tag)
    title('Estimator Selection: Residual Cost')
    legend('H_{11}','H_{12}','noise','Location','best'), ylim([0,intv(2)*5])
pause

%% TEST 3: CHI-SQUARES
% Chi^2 test of absolute modeling error
[confid,var,tag] = chi2test(X,Y,freq,FRFs,sCR,SYS);
figure
subplot(211),loglog(freq,confid(:,:,1)), hold on, loglog(freq,var(:,1),'k')
    legend([tag(:,1);'crlb'],'Location','best');
subplot(212),loglog(freq,confid(:,:,2)), hold on, loglog(freq,var(:,2),'k')
    legend([tag(:,2);'crlb'],'Location','best');
    
