function [lags,corr,cb50,frac50,tag,cb95,frac95] = residtest(x,y,freq,FRF,SYS,sCR,fs)
% RESID - Identification whiteness residuals test (MIMO).
%   [lags,corr,cb50,frac50,tag,cb95,frac95] = residtest(x,y,freq,FRF,SYS,sCR,fs)
% x,y,freq  : Input & output time domain data
% FRF,SYS   : FRF measurement & structured estim models
% sCR       : Cramer-Rao lower bound of measurement
% lags,corr : frequency lags & auto_corr matrix
% cb50/95   : Model residual 50/95%-confindence bounds
% frac,tag  : fraction under confidence bounds & model name
% Author    : Thomas Beauduin, KULeuven, PMA division, 2014
%%%%%
nroff = length(freq);               % number of frequency lines
nrofi = size(x,2);                  % number of inputs vectors
nrofo = size(y,2);                  % number of output vectors
nrofh = nrofi*nrofo;                % number of transfer functions
nrofm = length(fieldnames(SYS));    % number of system models
lags = (-nroff+1:nroff-1);          % whiteness frequency lags
nrofp = length(x(:,1))/fs*(freq(2)-freq(1));

% Calculation of scaling parameters
scale0 = (nrofp-2)/(nrofp-1);
scale = (nrofp-5/3)/(nrofp-11/12);
cb_scale0 = scale0*((nrofp-1)^(3/2)/(nrofp-2)/(nrofp-3)^(1/2));
cb_scale = scale*(nrofp-1)/(nrofp-2);
cb_scale = cb_scale*ones(size(lags));
cb_scale(nroff) = cb_scale0;
ac_scale = scale*ones(size(lags));
ac_scale(nroff) = scale0;

% Calculation of confidence Bounds
p50 = sqrt(-log(1-0.5)); p95 = sqrt(-log(1-0.95));
conf_bound = repmat(cb_scale./(nroff-abs(lags)).^(0.5),[2,1]);
cb50 = p50*conf_bound(1,:);
cb95 = p95*conf_bound(2,:);

% Calculation of auto-correlation & fraction
SYS_c = struct2cell(SYS);
select = (1:2*nroff-1); 
select(nroff) = [];
tag = cell(nrofm,nrofh);
frac50 = zeros(nrofm,nrofh);
frac95 = zeros(nrofm,nrofh);
corr = zeros(length(lags),nrofm,nrofh);
for h=1:nrofh
    i = ceil(h/nrofo); o = h-(i-1)*nrofo;
    for m = 1:nrofm
        FRFsys = squeeze(freqresp(SYS_c{m}(o,i),freq*2*pi));
        res = (FRF(:,h)-FRFsys)./sCR(:,h).^0.5;
        auto_corr = local_acorr_unbiased(res).*ac_scale;
        frac50(m,h) = length(find(abs(auto_corr(select)) - ...
                              cb50(select) > 0)) / (2*nroff-2)*100;
        frac95(m,h) = length(find(abs(auto_corr(select)) - ...
                              cb95(select) > 0)) / (2*nroff-2)*100;
        corr(:,m,h) = squeeze(abs(auto_corr));
    end
   [frac50(:,h),index] = sort(frac50(:,h),'descend');
    frac95(:,h) = frac95(index,h);
    corr(:,:,h) = corr(:,index,h);
    tag(:,h) = fieldnames(SYS);
    tag(:,h) = tag(index,h);
end

end

% ===== local helper (replaces xcorr -> no Signal Processing Toolbox) =========
function r = local_acorr_unbiased(v)
% Unbiased autocorrelation of vector V (FFT-based), returned as a ROW vector
% ordered for lags -(N-1):(N-1). Drop-in for xcorr(v,'unbiased') (matches to
% machine precision; only abs(.) is used downstream). No toolbox required.
v    = v(:);
N    = numel(v);
nfft = 2*N-1;                          % exact: linear == circular autocorr
a    = ifft(abs(fft(v,nfft)).^2);      % a(1)=lag0, a(k+1)=lag +k, a(nfft-k+1)=lag -k
r    = [a(N+1:nfft); a(1:N)].';        % reorder to lags -(N-1)..(N-1), as ROW
r    = r ./ (N - abs(-(N-1):(N-1)));   % 'unbiased' scaling
end
