function t = y(x,w)
[m, q] = size(w);
t = 0;
for i = 1:m
   t = t +(w(i)*(x.^(i-1)));
end

end