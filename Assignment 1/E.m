function error = E(w,x,t,lambda)
if ~exist('lambda','var')
 % third parameter does not exist, so default it to something
  lambda = 0;
end

[N, q] = size(x);

yn = y(x,w); 
error = 0.5*(sum((yn-t).^2)); 

error = error + 0.5*lambda*sum(w.^2); 

error = sqrt(2*error/N);


end