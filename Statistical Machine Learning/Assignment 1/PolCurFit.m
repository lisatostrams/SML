function w = PolCurFit(D_N,M,lambda)
if ~exist('lambda','var')
 % third parameter does not exist, so default it to something
  lambda = 0;
end

M = M+1;
x = D_N(1,:);
t = D_N(2,:);

A = zeros(M,M);
for i = 1:M
    for j = 1:M
        delta = i==j;
        A(i,j) = sum(x.^(i+j-2)) + lambda*delta;
    end
end

T = zeros(M,1);
for i = 1:M
    T(i) = sum(t.*(x.^(i-1)));
end

w = A\T;


end