
it_x = @(x)(x - (cos(x)/(-sin(x))));
xs =zeros(1,10);
xs(1)=1;
for i = 2:10
    xs(i)  = it_x(xs(i-1));
end

z = linspace(min(xs),2*pi,100);
hold on;
plot(z,sin(z))
plot(xs,sin(xs),'r.')


%%
phi = [0.3,0.44,0.46,0.6];
phi = [ones(1,size(phi,2));phi];
t = [1,0,1,0];
w = zeros(2,10); 
w(:,1) = [1;1];
sigmoid = @(w,x)(1/(1+exp(-(w'*x))));

for i = 2:10
    y_n = arrayfun(@(x1,x2)sigmoid(w(:,i-1),[x1;x2]),phi(1,:),phi(2,:));
    R = diag(y_n.*(1-y_n));
    w(:,i) = w(:,i-1) - inv(phi*R*phi')*(phi*(y_n-t)')
end
