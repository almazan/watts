function [Wx, Wy, r] = cca(X,Y, reg1, reg2)

% CCA calculate canonical correlations
%
% [Wx Wy r] = cca(X,Y) where Wx and Wy contains the canonical correlation
% vectors as columns and r is a vector with corresponding canonical
% correlations. The correlations are sorted in descending order. X and Y
% are matrices where each column is a sample. Hence, X and Y must have
% the same number of columns.
%
% Example: If X is M*K and Y is N*K there are L=MIN(M,N) solutions. Wx is
% then M*L, Wy is N*L and r is L*1.
%
%
% ? 2000 Magnus Borga, Link?pings universitet

if nargin < 4    
    reg2=reg1;
end

% --- Calculate covariance matrices ---

N = size(X, 1);
Dx = size(X, 2);
Dy = size(Y, 2);    


Cxx = X'*X / N + reg1*eye(Dx);
Cyy = Y'*Y/N + reg2*eye(Dy);
%Cyy = reg*eye(Dy);
Cxy = X'*Y / N;
Cyx = Cxy';

%% Rewrite...
% z = [X;Y];
% C = cov(z.');
% sx = size(X,1);
% sy = size(Y,1);
% Cxx = cov(X)
% Cxx = C(1:sx, 1:sx) + 10^(-8)*eye(sx);
% Cxy = C(1:sx, sx+1:sx+sy);
% Cyx = Cxy';
% Cyy = C(sx+1:sx+sy, sx+1:sx+sy) + 10^(-8)*eye(sy);
%%% ...


%invCyy = inv(Cyy);
%invCxx = inv(Cxx);
% --- Calcualte Wx and r ---

%M =  invCxx*Cxy * invCyy * Cyx;
M =  ((Cxx\Cxy)/Cyy)*Cyx;

[Wx,r] = eig(M); % Basis in X
r = sqrt(real(r));      % Canonical correlations

% --- Sort correlations ---
    
V = fliplr(Wx);		% reverse order of eigenvectors
r = flipud(diag(r));	% extract eigenvalues anr reverse their orrer
[r,I]= sort((real(r)));	% sort reversed eigenvalues in ascending order
r = flipud(r);		% restore sorted eigenvalues into descending order
for j = 1:length(I)
  Wx(:,j) = V(:,I(j));  % sort reversed eigenvectors in ascending order
end
Wx = fliplr(Wx);	% restore sorted eigenvectors into descending order



% --- Calcualte Wy  ---

Wy = (Cyy\Cyx)*Wx;     % Basis in Y
Wy = Wy./repmat(sqrt(sum(abs(Wy).^2)),Dy,1); % Normalize Wy