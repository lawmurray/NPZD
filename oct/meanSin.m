function A = meanSin(hyp, x, i)

% Sin mean function. The mean function is parameterized as:
%
% m(x) = sum_i (a_i*sin(2*pi*(x_i - phi_i)/T_i))
%
% that is, each T_i gives the period and each phi_i the phase, both in days.
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
%         phi_D
%         log(T_1)
%         log(T_2)
%         ..
%         log(T_D) ]
%
% See also MEANFUNCTIONS.M.

if nargin<2, A = '3*D'; return; end             % report number of hyperparameters 
[n,D] = size(x);
if any(size(hyp)~=[3*D,1]), error('Exactly 3*D hyperparameters needed.'), end

a = hyp(1:D);
phi = hyp((D+1):(2*D));
T = exp(hyp((2*D+1):(3*D)));

if nargin==2  
  A = sin(2*pi*(x - repmat(phi', n, 1))./repmat(T', n, 1))*a;
else
  j = mod(i, D) + 1;
  if i <= D
    A = sin(2*pi*(x(:,j) - phi(j))./T(j));
  elseif i <= 2*D
    A = cos(2*pi*((x(:,j) - phi(j))/T(j)))*((-2*pi)/T(j))*a(j);
  elseif i <= 3*D
    A = cos(2*pi*((x(:,j) - phi(j)/T(j))))*((2*pi*phi(j)*T(j).^-2))*a(j);
  else
    A = zeros(n,1);
  end
end
