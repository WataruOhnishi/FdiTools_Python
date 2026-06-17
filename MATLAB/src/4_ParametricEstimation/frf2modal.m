function [modal, Pm] = frf2modal(Pest, nrbm, nflex, varargin)
%FRF2MODAL - structured modal identification of MIMO systems from FRF data.
%   [modal, Pm] = frf2modal(Pest, nrbm, nflex)
%   [modal, Pm] = frf2modal(Pest, nrbm, nflex, 'Name', value, ...)
%
% Estimates a modal model (PROPORTIONAL or general-VISCOUS damping) from a
% measured FRF matrix, following the two-stage, structured (rank-one) approach
%   M. van der Hulst, R.A. Gonzalez, K. Classens, P. Tacx, N. Dirkx,
%   J. van de Wijdeven, T. Oomen, "Structured identification of multivariable
%   modal systems", Mech. Syst. Signal Process. 247 (2026) 113948.
%
% Models:
%  proportional (eqs.(11)-(13)) - real, rank-one residues:
%    P(s) = D + sum_j phi_l^rbm_j phi_r^rbm,T_j / s^2
%             + sum_i phi_l,i phi_r,i^T / (s^2 + 2 zeta_i w_i s + w_i^2)
%  general (eqs.(2),(6),(46)) - complex mode shapes psi_l,psi_r, with
%    L_i = psi_l,i psi_r,i^T (rank one, complex), per flexible mode:
%    L_i/(s-lambda_i) + conj(L_i)/(s-conj(lambda_i)),
%    lambda_i = -zeta_i w_i + j w_i sqrt(1-zeta_i^2).
%
% Two stages: (1) additive model with FREE residue matrices (poles by LM,
% residues by per-I/O linear LS, eqs.(21),(23)); (2) rank-one SVD projection
% of each residue (Eckart-Young, sec.5.1) followed by a structured refinement
% against the FRF (sec.5.2).
%
% INPUTS
%   Pest   : measured FRF as an frd (ny x nu x N), e.g. from TIME2FRF_ML (MIMO)
%   nrbm   : number of rigid-body modes ; nflex : number of flexible modes
% OPTIONS
%   'damping'    : 'proportional' (default) | 'general'
%   'initfreq'   : nflex initial modal frequencies [Hz] (default: CMIF peaks)
%   'initdamp'   : initial damping ratio (scalar or nflex), default 0.01
%   'feedthrough': estimate a constant D term {true}|false
%   'weight'     : 'invmag' (W = 1/|G|, eq.(52)) {default} | 'none'
%   'band'       : [fl fh] frequency band used for the fit (default: all)
%   'maxiter'    : max LM iterations per stage (default 100)
%   'tol'        : relative parameter tolerance (default 1e-8)
% OUTPUTS
%   modal : struct with the modal parameters (see below) ; .damping set
%   Pm    : identified modal model as an ss object (real, minimal)
%
% See also TIME2FRF_ML, FRD.
%   Author : Wataru Ohnishi, The University of Tokyo, 2026

% ---- options ------------------------------------------------------------
opt = struct('damping','proportional', 'initfreq',[], 'initdamp',0.01, ...
             'feedthrough',true, 'weight','invmag', 'band',[], ...
             'maxiter',100, 'tol',1e-8);
for k = 1:2:numel(varargin), opt.(lower(varargin{k})) = varargin{k+1}; end
gen = strcmpi(opt.damping,'general');

% ---- extract FRF --------------------------------------------------------
G = Pest.ResponseData;  f = Pest.Frequency(:);
if ~isempty(opt.band), s_=f>=opt.band(1)&f<=opt.band(2); G=G(:,:,s_); f=f(s_); end
[ny,nu,N] = size(G);
w = 2*pi*f;  xi = 1i*w;
switch lower(opt.weight)
    case 'invmag', Wt = 1./max(abs(G),eps);
    otherwise,     Wt = ones(ny,nu,N);
end

% ---- initial poles ------------------------------------------------------
if isempty(opt.initfreq), wn0 = 2*pi*cmif_peaks(G,f,nflex);
else,                     wn0 = 2*pi*opt.initfreq(:); end
ze0 = opt.initdamp(:); if isscalar(ze0), ze0 = ze0*ones(nflex,1); end

% =====================================================================
% STAGE 1: additive model (variable projection over the poles)
% =====================================================================
p0  = [wn0; ze0];
r1  = @(p) stage1_residual(p, G, xi, Wt, nrbm, nflex, opt.feedthrough, gen);
p1  = lmsolve(r1, p0, opt.maxiter, opt.tol);
[~, C] = stage1_residual(p1, G, xi, Wt, nrbm, nflex, opt.feedthrough, gen);
wn = p1(1:nflex);  ze = p1(nflex+1:end);
[Dest, Rrbm_full, B0, B1] = unpack_residues(C, ny, nu, nrbm, nflex, opt.feedthrough, gen);

% =====================================================================
% rank-one SVD initialization of the mode shapes (sec. 5.1)
% =====================================================================
phil_r = zeros(ny,nrbm); phir_r = zeros(nu,nrbm);
if nrbm > 0
    [U,S,V] = svd(Rrbm_full);
    for j = 1:nrbm, phil_r(:,j)=U(:,j)*sqrt(S(j,j)); phir_r(:,j)=V(:,j)*sqrt(S(j,j)); end
end
if gen
    psil = zeros(ny,nflex); psir = zeros(nu,nflex);   % complex mode shapes
    for i = 1:nflex
        lam = -ze(i)*wn(i) + 1i*wn(i)*sqrt(max(1-ze(i)^2,0));
        Li  = (B0(:,:,i) + lam*B1(:,:,i))/(lam - conj(lam));   % eq.(40)
        [U,S,V] = svd(Li);
        psil(:,i) = U(:,1)*sqrt(S(1,1));
        psir(:,i) = conj(V(:,1))*sqrt(S(1,1));                 % L = psil*psir.'
    end
else
    phil = zeros(ny,nflex); phir = zeros(nu,nflex);   % real mode shapes
    for i = 1:nflex
        [U,S,V] = svd(B0(:,:,i));
        phil(:,i)=U(:,1)*sqrt(S(1,1)); phir(:,i)=V(:,1)*sqrt(S(1,1));
    end
end

% =====================================================================
% STAGE 2: refine the rank-one modal model against the FRF (sec. 5.2)
% =====================================================================
if gen
    rho0 = [wn(:); ze(:); real(psil(:)); imag(psil(:)); real(psir(:)); imag(psir(:)); ...
            phil_r(:); phir_r(:)];
else
    rho0 = [wn(:); ze(:); phil(:); phir(:); phil_r(:); phir_r(:)];
end
if opt.feedthrough, rho0 = [rho0; Dest(:)]; end
r2  = @(r) stage2_residual(r, G, xi, Wt, ny, nu, nrbm, nflex, opt.feedthrough, gen);
rho = lmsolve(r2, rho0, opt.maxiter, opt.tol);

% ---- unpack & normalize -------------------------------------------------
[wn, ze, mshapes, phil_r, phir_r, Dest] = unpack_modal(rho, ny, nu, nrbm, nflex, opt.feedthrough, gen);
modal.damping = opt.damping;
modal.wn   = wn/(2*pi);  modal.zeta = ze;
modal.phil_rbm = phil_r; modal.phir_rbm = phir_r;
modal.Rrbm = zeros(ny,nu,nrbm);
for j = 1:nrbm, modal.Rrbm(:,:,j) = phil_r(:,j)*phir_r(:,j).'; end
modal.D = Dest;
if gen
    psil = mshapes.psil; psir = mshapes.psir;
    for i = 1:nflex                                   % scale: unit-norm psir, phase
        a = norm(psir(:,i)); if a>0, psir(:,i)=psir(:,i)/a; psil(:,i)=psil(:,i)*a; end
        [~,im] = max(abs(psir(:,i))); ph = angle(psir(im,i));
        psir(:,i)=psir(:,i)*exp(-1i*ph); psil(:,i)=psil(:,i)*exp(1i*ph);
    end
    modal.psil = psil; modal.psir = psir;
    modal.lambda = -ze(:).*(2*pi*modal.wn(:)) + 1i*(2*pi*modal.wn(:)).*sqrt(max(1-ze(:).^2,0));
else
    phil = mshapes.phil; phir = mshapes.phir;
    for i = 1:nflex
        a = norm(phir(:,i)); if a>0, phir(:,i)=phir(:,i)/a; phil(:,i)=phil(:,i)*a; end
        if phil(1,i)<0, phil(:,i)=-phil(:,i); phir(:,i)=-phir(:,i); end
    end
    modal.phil = phil; modal.phir = phir;
end
Pm = modal2ss(modal, nrbm, nflex, gen);
end

% =====================================================================
% ===== local functions ===============================================
% =====================================================================
function [r, C] = stage1_residual(p, G, xi, Wt, nrbm, nflex, hasD, gen)
[ny,nu,N] = size(G);
wn = p(1:nflex); ze = p(nflex+1:end);
B  = basis(xi, wn, ze, nrbm, hasD, gen);
Nb = size(B,2);  C = zeros(Nb,ny,nu);  r = zeros(2*ny*nu*N,1); idx=0;
for o=1:ny
    for u=1:nu
        g=squeeze(G(o,u,:)); sw=sqrt(squeeze(Wt(o,u,:)));
        Bw=B.*sw; c=Bw\(g.*sw); C(:,o,u)=c;
        e=(g-B*c).*sw; r(idx+(1:N))=real(e); r(idx+N+(1:N))=imag(e); idx=idx+2*N;
    end
end
end

function B = basis(xi, wn, ze, nrbm, hasD, gen)
N=numel(xi); cols={};
if hasD,   cols{end+1}=ones(N,1);   end
if nrbm>0, cols{end+1}=1./(xi.^2);  end
for i=1:numel(wn)
    Ai = xi.^2 + 2*ze(i)*wn(i)*xi + wn(i)^2;
    cols{end+1}=1./Ai;                          %#ok<AGROW>  % N_0 term
    if gen, cols{end+1}=xi./Ai; end             %#ok<AGROW>  % N_1 s term
end
B=[cols{:}];
end

function [D, Rrbm, B0, B1] = unpack_residues(C, ny, nu, nrbm, nflex, hasD, gen)
D=zeros(ny,nu); Rrbm=zeros(ny,nu); B0=zeros(ny,nu,nflex); B1=zeros(ny,nu,nflex);
row=0;
if hasD,   for o=1:ny,for u=1:nu, D(o,u)=C(row+1,o,u);    end,end, row=row+1; end
if nrbm>0, for o=1:ny,for u=1:nu, Rrbm(o,u)=C(row+1,o,u); end,end, row=row+1; end
for i=1:nflex
    for o=1:ny,for u=1:nu, B0(o,u,i)=C(row+1,o,u); end,end, row=row+1;
    if gen, for o=1:ny,for u=1:nu, B1(o,u,i)=C(row+1,o,u); end,end, row=row+1; end
end
end

function r = stage2_residual(rho, G, xi, Wt, ny, nu, nrbm, nflex, hasD, gen)
[wn, ze, ms, phil_r, phir_r, D] = unpack_modal(rho, ny, nu, nrbm, nflex, hasD, gen);
N=numel(xi); P=repmat(D,[1 1 N]);
for j=1:nrbm
    Rj=phil_r(:,j)*phir_r(:,j).'; P=P+reshape(Rj(:)*(1./(xi.^2)).',[ny nu N]);
end
for i=1:nflex
    w=wn(i); z=ze(i);
    if gen
        lam=-z*w+1i*w*sqrt(max(1-z^2,0)); Li=ms.psil(:,i)*ms.psir(:,i).';
        di = 1./(xi-lam) ; dc = 1./(xi-conj(lam));
        P = P + reshape(Li(:)*di.' + conj(Li(:))*dc.', [ny nu N]);
    else
        Ri=ms.phil(:,i)*ms.phir(:,i).'; di=1./(xi.^2+2*z*w*xi+w^2);
        P = P + reshape(Ri(:)*di.', [ny nu N]);
    end
end
E=(G-P).*sqrt(Wt); r=[real(E(:)); imag(E(:))];
end

function [wn, ze, ms, phil_r, phir_r, D] = unpack_modal(rho, ny, nu, nrbm, nflex, hasD, gen)
i0=0;
wn=rho(i0+(1:nflex)); i0=i0+nflex;
ze=rho(i0+(1:nflex)); i0=i0+nflex;
if gen
    rl=reshape(rho(i0+(1:ny*nflex)),[ny nflex]); i0=i0+ny*nflex;
    il=reshape(rho(i0+(1:ny*nflex)),[ny nflex]); i0=i0+ny*nflex;
    rr=reshape(rho(i0+(1:nu*nflex)),[nu nflex]); i0=i0+nu*nflex;
    ir=reshape(rho(i0+(1:nu*nflex)),[nu nflex]); i0=i0+nu*nflex;
    ms.psil=rl+1i*il; ms.psir=rr+1i*ir;
else
    ms.phil=reshape(rho(i0+(1:ny*nflex)),[ny nflex]); i0=i0+ny*nflex;
    ms.phir=reshape(rho(i0+(1:nu*nflex)),[nu nflex]); i0=i0+nu*nflex;
end
phil_r=reshape(rho(i0+(1:ny*nrbm)),[ny nrbm]); i0=i0+ny*nrbm;
phir_r=reshape(rho(i0+(1:nu*nrbm)),[nu nrbm]); i0=i0+nu*nrbm;
if hasD, D=reshape(rho(i0+(1:ny*nu)),[ny nu]); else, D=zeros(ny,nu); end
end

function Pm = modal2ss(modal, nrbm, nflex, gen)
% Real minimal modal realization (per-mode blocks; eq.(65)/Appendix A.3).
if gen, ny=size(modal.psil,1); nu=size(modal.psir,1);
else,   ny=size(modal.phil,1); nu=size(modal.phir,1); end
A=[]; B=zeros(0,nu); C=zeros(ny,0);
for j=1:nrbm                                   % rigid: phi_l phi_r^T / s^2
    A=blkdiag(A,[0 1;0 0]);
    B=[B; zeros(1,nu); modal.phir_rbm(:,j).'];
    C=[C, modal.phil_rbm(:,j), zeros(ny,1)];
end
for i=1:nflex
    w=2*pi*modal.wn(i); z=modal.zeta(i);
    if gen
        lam=modal.lambda(i);
        A=blkdiag(A,[real(lam) -imag(lam); imag(lam) real(lam)]);
        B=[B; real(modal.psir(:,i)).'; imag(modal.psir(:,i)).'];
        C=[C, 2*real(modal.psil(:,i)), -2*imag(modal.psil(:,i))];
    else
        A=blkdiag(A,[0 1; -w^2 -2*z*w]);
        B=[B; zeros(1,nu); modal.phir(:,i).'];
        C=[C, modal.phil(:,i), zeros(ny,1)];
    end
end
Pm=ss(A,B,C,modal.D);
end

function fpk = cmif_peaks(G, f, nflex)
N=numel(f); sv=zeros(N,1);
for k=1:N, sv(k)=svds(G(:,:,k),1); end
lsv=20*log10(sv);
pk=find(lsv(2:end-1)>lsv(1:end-2)&lsv(2:end-1)>lsv(3:end))+1;
[~,ord]=sort(lsv(pk),'descend'); pk=pk(ord);
if numel(pk)<nflex
    fpk=logspace(log10(f(2)),log10(f(end)),nflex+2).'; fpk=fpk(2:end-1);
else, fpk=sort(f(pk(1:nflex))); end
end

function x = lmsolve(resfun, x0, maxiter, tol)
x=x0(:); r=resfun(x); cost=r.'*r; mu=1e-3;
for it=1:maxiter
    J=numjac(resfun,x,r); H=J.'*J; g=J.'*r;
    while true
        dx=-(H+mu*diag(max(diag(H),1e-12)))\g;
        xn=x+dx; rn=resfun(xn); cn=rn.'*rn;
        if cn<cost
            x=xn; r=rn; mu=max(mu/3,1e-12);
            if norm(dx)<=tol*(norm(x)+tol), return; end
            cost=cn; break
        else
            mu=mu*3; if mu>1e12, return; end
        end
    end
end
end

function J = numjac(resfun, x, r0)
n=numel(x); m=numel(r0); J=zeros(m,n);
for k=1:n
    dk=1e-6*max(abs(x(k)),1e-6); xp=x; xp(k)=xp(k)+dk;
    J(:,k)=(resfun(xp)-r0)/dk;
end
end
