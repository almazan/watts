% This function normalize a set of vectors 
% Parameters:
%   v     the set of vectors to be normalized (column stored)
%   nr    the norm for which the normalization is performed (Default: Euclidean)
%
% Output:
%   vout  the normalized vector
%   vnr   the norms of the input vectors
%
% Remark: the function return Nan for vectors of null norm
function [vout, vnr] = yael_fvecs_normalize (v, nr)

fprintf ('# Warning: consider using the Mex implementation instead of this pure Matlab one\n');

if nargin < 2
  nr = 2;
end

% norm of each column
vnr = (sum (v.^nr)) .^ (1 / nr);

% sparse multiplication to apply the norm
vout = single (double (v) * sparse (diag (double (1 ./ vnr))));
