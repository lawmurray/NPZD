function K = covSin(hyp, x, z, i)

% Stationary covariance function for a smooth periodic function, with period 365:
%
% k(x,y) = sf2 * exp( -2*sin^2( pi*||x-y||/365 )/ell^2 )
%
% where the hyperparameters are:
%
% hyp = [ log(ell)
%         log(sqrt(sf2)) ]
%
% Adapted from GPML covPeriodic function.

if nargin<2, K = '2'; return; end                  % report number of parameters
if nargin<3, z = []; end                                   % make sure, z exists
xeqz = numel(z)==0; dg = strcmp(z,'diag') && numel(z)>0;        % determine mode

n = size(x,1);
ell = exp(hyp(1));
p   = 365.0;
sf2 = exp(2*hyp(2));

% precompute distances
if dg                                                               % vector kxx
  K = zeros(size(x,1),1);
else
  if xeqz                                                 % symmetric matrix Kxx
    K = sqrt(sq_dist(x'));
  else                                                   % cross covariances Kxz
    K = sqrt(sq_dist(x',z'));
  end
end

K = pi*K/p;
if nargin<4                                                        % covariances
    K = sin(K)/ell; K = K.*K; K =   sf2*exp(-2*K);
else                                                               % derivatives
  if i==1
    K = sin(K)/ell; K = K.*K; K = 4*sf2*exp(-2*K).*K;
  elseif i==2
    K = sin(K)/ell; K = K.*K; K = 2*sf2*exp(-2*K);
  else
    error('Unknown hyperparameter')
  end
end
