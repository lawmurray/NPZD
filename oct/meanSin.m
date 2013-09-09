function A = meanSin(hyp, x, i)

% Sin mean function with period of 365. The mean function is parameterized as:
%
% m(x) = sum_i (a_i*sin(2*pi*(x_i - phi_i)/365))
%
% that is, each phi_i givesn the phase.
%
% The hyperparameter is:
%
% hyp = [ a_1
%         a_2
%         ..
%         a_D
%         phi_1
%         phi_2
%         ..
%         phi_D ]
%
% See also MEANFUNCTIONS.M.

if nargin<2, A = '2*D'; return; end             % report number of hyperparameters 
[n,D] = size(x);
if any(size(hyp)~=[2*D,1]), error('Exactly 2*D hyperparameters needed.'), end

a = hyp(1:D);
phi = hyp((D+1):(2*D));
T = 365.0;

if nargin==2  
  A = sin(2*pi*(x - repmat(phi', n, 1))/365.0)*a;
else
  j = mod(i, D) + 1;
  if i <= D
    A = sin(2*pi*(x(:,j) - phi(j))./T);
  elseif i <= 2*D
    A = cos(2*pi*((x(:,j) - phi(j))/T))*((-2*pi)/T)*a(j);
  else
    A = zeros(n,1);
  end
end
