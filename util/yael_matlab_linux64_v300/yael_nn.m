% Return the k nearest neighbors of a set of query vectors
%
% Usage: [ids,dis] = nn(v, q, k, distype)
%   v                the dataset to be searched (one vector per column)
%   q                the set of queries (one query per column)
%   k  (default:1)   the number of nearest neigbors we want
%   distype          distance type: 1=L1, 
%                                   2=L2         -> Warning: return the square L2 distance
%                                   3=chi-square -> Warning: return the square Chi-square
%                                   4=signed chis-square
%                                   16=cosine    -> Warning: return the *smallest* cosine values
%                                                   Use -query to obtain the largest
%                    available in Mex-version only
%
% Returned values
%   idx         the vector index of the nearest neighbors
%   dis         the corresponding *square* distances
%
% Both v and q contains vectors stored in columns, so transpose them if needed
function [idx, dis] = yael_nn (v, q, k)


if narg >= 4
  error ('yael_nn is defined for arbitrary distances only in the Mex version\nonsider using the Mex-compiled version instead\n');
end
  
fprintf ('* Warning: this is the slow version of nn\nConsider using the Mex-compiled version instead\n');

if nargin < 3, k = 1; end

assert (size (v, 1) == size (q, 1));

n = size (v, 2);
nq = size (q, 2);

v_nr = sum (v.^2);
q_nr = sum (q.^2);
dis = repmat (v_nr', 1, nq) + repmat (q_nr, n, 1) - 2 * v' * q;
[dis, idx] = sort (dis);

dis = dis (1:k, :);
idx = idx (1:k, :);
