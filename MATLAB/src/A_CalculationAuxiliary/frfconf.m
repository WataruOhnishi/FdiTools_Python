function f = frfconf(p, M)
%FRFCONF - confidence-radius factor for a measured FRF.
%   f = frfconf(p, M)
%
% The 100*p% circular confidence bound on an FRF measured from M averaged
% periods has radius   radius_p = sigma_Ghat * f   (Pintelon-Schoukens 2012
% eq.(2-40)), where sigma_Ghat is the FRF standard deviation (UserData.sG).
%
%   f = sqrt( F_p(2, 2M-2) ).
%
% The numerator d.o.f. is 2 (real + imaginary part of the FRF), so the
% F-quantile has the closed form  F_p(2,nu) = (nu/2)((1-p)^(-2/nu) - 1),
% nu = 2M-2. For M < 2 the large-M limit sqrt(-log(1-p)) (eq.(2-31)) is used.
%
%   bode_fdi(Pest, 'unc','sG', 'style','band', 'sigma', frfconf(0.95, Pest.UserData.nrofp));
%
% See also TIME2FRF_ML, BODE_FDI.
%   Author : Wataru Ohnishi, The University of Tokyo, 2026

if nargin < 2, M = []; end
if p <= 0 || p >= 1
    error('frfconf:p','p must be in the open interval (0,1).');
end

if isempty(M) || ~isfinite(M) || M < 2
    f = sqrt(-log(1-p));
else
    nu = 2*M - 2;
    f  = sqrt( (nu/2) * ((1-p)^(-2/nu) - 1) );
end
end
