function [G0, wnHz, zeta, phi] = mimobench()
%MIMOBENCH - 2-input/2-output rank-one proportionally-damped modal benchmark.
% Shared truth model for the Step_MIMO* examples. Each flexible mode is a
% rank-one (reciprocal) residue phi*phi^T over a 2nd-order denominator, so it
% matches the structure targeted by FRF2MODAL.
%   [G0, wnHz, zeta, phi] = mimobench()
%   G0   : 2x2 ss model (force -> position-like)
%   wnHz : modal frequencies [Hz]
%   zeta : modal damping ratios
%   phi  : 2 x 3 output mode shapes (reciprocal: input shape = output shape)
%   Author : Wataru Ohnishi, The University of Tokyo, 2026
s    = tf('s');
wnHz = [40; 95; 180];                 % modal frequencies [Hz]
zeta = [0.010; 0.015; 0.020];         % damping ratios
phi  = [1.0  1.0  1.0;                 % 2 outputs x 3 modes
        0.6 -0.8  0.4];
g    = [1.0  0.7  0.5];                % modal gains (relative)
% scale the gains so the (1,1) element has a 0 dB DC gain
dc11 = sum( g(:) .* (phi(1,:).').^2 ./ (2*pi*wnHz).^2 );
g    = g / dc11;
G0 = tf(zeros(2,2));
for i = 1:numel(wnHz)
    w  = 2*pi*wnHz(i);
    Ri = g(i)*phi(:,i)*phi(:,i).';     % rank-one symmetric residue
    G0 = G0 + Ri*(1/(s^2 + 2*zeta(i)*w*s + w^2));
end
G0 = ss(G0);
end
