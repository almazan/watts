function Wx = learn_regression(X,Y,reg)
N = size(X,1);
D = size(X,2);
XX = X'*X + eye(D)*reg;
XY = X'*Y;
Wx = XX\XY;
