function varargout = time2frf_ml(varargin)
%TIME2FRF_ML - maximum-likelihood estimation of FRF (SISO/SIMO matrix core).
% MIMO is handled by the iodata method @iodata/time2frf_ml.m.

% Two calling conventions (selected by the number of inputs):
%   Pest = time2frf_ml(x, y, ms)                                  % returns an frd
%   [Xs,Ys,FRFs,FRFn,freq,sX2,sY2,cXY,sCR] = time2frf_ml(x,y,fs,fl,fh,df)
%
% x,y  : periodic measurement (samples x channels)
% ms   : multisine from multisine.m (uses ms.harm.fs/fl/fh/df and ms.options)
% Pest : FRF (frd). UserData fields:
%   X,Y    : period-averaged spectra Xbar,Ybar
%   FRFs   : Ghat = Ybar/Xbar ;  FRFn : noise FRF (from non-excited lines)
%   sX2,sY2: per-component variance of Xbar,Ybar (variance of the mean, 1/M incl.)
%   cXY    : per-component covariance of Xbar,Ybar
%   sCR    : per-component FRF std (= sigma_Ghat/sqrt2; chi^2 / residual tests)
%   sG     : FRF std sigma_Ghat (PS2012 eq.2-38, = sqrt2*sCR)
%   nrofp  : number of averaged periods M (for FRFCONF confidence bounds)
%   ms     : the multisine
% Author : Thomas Beauduin (KU Leuven, 2014) / Wataru Ohnishi (UTokyo, 2019)
%%%%%

if length(varargin) < 5 % structured input
    x = varargin{1};
    y = varargin{2};
    ms = varargin{3};
    if nargin == 3
        flagTime = false;
    else
        flagTime = varargin{4};
    end
    % decompose structure
    fs = ms.harm.fs; fl = ms.harm.fl; fh = ms.harm.fh; df = ms.harm.df;
    
else % FdiTools classical input
    x = varargin{1};
    y = varargin{2};
    fs = varargin{3};
    fl = varargin{4};
    fh = varargin{5};
    df = varargin{6};
end


[~,nrofi] = size(x);                                % number of inputs
[~,nrofo] = size(y);                                % number of outputs
nrofh = nrofi*nrofo;                                % number of tf's (Hxy)
nrofs = fs/df;                                      % samples per period
nl = ceil(fl/df); nh = floor(fh/df);                % low & high freq
freq = double((nl:1:nh)'/(nrofs/fs));               % full freq lines
nroff = length(freq);                               % number of freq lines
nrofp = double(floor(length(x)/nrofs));             % number of period

% Calculation of signal fft data
INP = zeros(nroff,nrofp,nrofi);
OUT = zeros(nroff,nrofp,nrofo);
Xs = zeros(nroff,nrofi); sX2 = zeros(nroff,nrofi);
Ys = zeros(nroff,nrofo); sY2 = zeros(nroff,nrofo);
cXY = zeros(nroff,nrofh);
FRFs = zeros(nroff,nrofh);
sCR = zeros(nroff,nrofh);
sG = zeros(nroff,nrofh);
for i=1:nrofi
    for p=1:nrofp
        Ip = fft(x(1+(p-1)*nrofs:p*nrofs,i));       % fft of 1x period
        INP(:,p,i) = Ip(nl+1:nh+1);                 % fft dc-term removal
    end
    Xs(:,i) = mean(INP(:,:,i),2);                   % Xbar: sample mean spectrum
    sX2(:,i)=((std(INP(:,:,i),0,2)).^2)/2/nrofp;    % var of Re/Im of Xbar (var of the mean, per component, 1/M incl.)
end
for o=1:nrofo
    for p=1:nrofp
        Op = fft(y(1+(p-1)*nrofs:p*nrofs,o));
        OUT(:,p,o) = Op(nl+1:nh+1);
    end
    Ys(:,o) = mean(OUT(:,:,o),2);                   % Ybar: sample mean spectrum
    sY2(:,o)=((std(OUT(:,:,o),0,2)).^2)/2/nrofp;    % var of Re/Im of Ybar (var of the mean, per component)
end
for i=1:nrofi
    for o=1:nrofo
        for f=1:nroff
            Cf = cov(INP(f,:,i),OUT(f,:,o));        % measurement covariance
            cXY(f,(i-1)*nrofo+o) = Cf(1,2)/2/nrofp; % cov of Re/Im of Xbar,Ybar (per component)
        end
        FRFs(:,(i-1)*nrofo+o) = Ys(:,o)./Xs(:,i);   % Ghat = Ybar/Xbar
        % sCR: per-component (real/imag) FRF std = sigma_Ghat/sqrt2 (CR bound),
        %   sCR^2 = |Ghat|^2 (sX2/|Xbar|^2 + sY2/|Ybar|^2 - 2Re(cXY/(Xbar* Ybar))).
        sCR(:,(i-1)*nrofo+o) = ...
            sqrt(abs(FRFs(:,(i-1)*nrofo+o)).^2.*(sX2(:,i)./(abs(Xs(:,i))).^2 ...
            + sY2(:,o)./(abs(Ys(:,o))).^2 ...
            - 2*real(cXY(:,(i-1)*nrofo+o)./(conj(Xs(:,i)).*Ys(:,o)))));
        % sG: FRF std sigma_Ghat (PS2012 eq.2-38) = sqrt(2)*sCR.
        sG(:,(i-1)*nrofo+o) = sqrt(2)*sCR(:,(i-1)*nrofo+o);
    end
end

% Calculation of noise fft data
OUT = zeros(nroff*2,floor(nrofp/2),nrofo);
NSE = zeros(nroff,floor(nrofp/2),nrofo);
Yn = zeros(nroff,nrofo); FRFn = zeros(nroff,nrofh);
for o=1:nrofo
    for p=1:floor(nrofp/2)
        Op = fft(y(1+(p-1)*nrofs*2:p*nrofs*2,o));   % fft of 2x period
        OUT(:,p,o) = Op(2*nl:2*nh+1);               % fft dc-term removal
    end
    index = 1;
    for f=1:2:nroff*2                               % uneven freq lines
        NSE(index,:,o) = OUT(f,:,o);
        index = index + 1;
    end
    Yn(:,o) = mean(NSE(:,:,o),2);
end
for i=1:nrofi
    for o=1:nrofo
        FRFn(:,(i-1)*nrofo+o) = Yn(:,o)./Xs(:,i);   % noise frf calc
    end
end

if length(varargin) < 5 % structured i/o
    % delete data for qlog excitation
    if any(strcmp(ms.options.gtp,{'q','qlog'}))
        excond = ms.ex-(ms.ex(1)-1); freq = ms.freq(ms.ex);
        Xs = Xs(excond); Ys = Ys(excond,:); FRFs = FRFs(excond,:); FRFn = FRFn(excond,:);
        sX2 = sX2(excond,:); sY2 = sY2(excond,:); cXY = cXY(excond,:); sCR = sCR(excond,:);
        sG = sG(excond,:);
    end
    
    Pest = frd(FRFs(:,1),freq,'FrequencyUnit','Hz');
    % for SIMO model id
    if nrofo > 1
        for o=2:nrofo
            Pest = [Pest;frd(FRFs(:,o),freq,'FrequencyUnit','Hz');];
        end
    end
    if nrofi > 1
        Pest = frd(zeros(nrofo,nrofi,size(freq,1)),freq,'FrequencyUnit','Hz');
        for i=1:nrofi
            for o=1:nrofo
                Pest(o,i) = frd(FRFs(:,(i-1)*nrofo+o),freq,'FrequencyUnit','Hz');
            end
        end
    end
    Pest.UserData.X = Xs;
    Pest.UserData.Y = Ys;
    Pest.UserData.FRFn = FRFn;
    Pest.UserData.sX2 = sX2;
    Pest.UserData.sY2 = sY2;
    Pest.UserData.cXY = cXY;
    Pest.UserData.sCR = sCR;
    Pest.UserData.sG = sG;       % FRF standard deviation, PS2012 eq.(2-38)
    Pest.UserData.nrofp = nrofp;
    Pest.UserData.ms = ms;
    
    if flagTime % save Time domain data
        Pest.UserData.x = x;
        Pest.UserData.y = y;
    end
    
    if isfield(Pest.UserData,'x'), Pest = fdicohere(Pest); end
    varargout{1} = Pest;
else % FdiTools classical input
    varargout{1} = Xs;
    varargout{2} = Ys;
    varargout{3} = FRFs;
    varargout{4} = FRFn;
    varargout{5} = freq;
    varargout{6} = sX2;
    varargout{7} = sY2;
    varargout{8} = cXY;
    varargout{9} = sCR;
end

end
