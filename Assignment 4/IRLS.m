
it_x = @(x)(x - (cos(x)/(-sin(x))));
xs =zeros(1,5);
xs(1)=1;
for i = 2:5
    xs(i)  = it_x(xs(i-1));
end

z = linspace(min(xs),2*pi,100);
figure(1);
hold on;
plot(z,sin(z))
plot(xs,sin(xs),'r.')
hold off;

%%
%initialisation
phi = [0.3,0.44,0.46,0.6];
phi = [ones(1,size(phi,2));phi];
t = [1,0,1,0];
w = zeros(2,6); 
w(:,1) = [1;1];

%algorithm
sigmoid = @(w,x)(1/(1+exp(-(w'*x))));
for i = 2:6
    y_n = arrayfun(@(x1,x2)sigmoid(w(:,i-1),[x1;x2]),phi(1,:),phi(2,:));
    R = diag(y_n.*(1-y_n));
    w(:,i) = w(:,i-1) - inv(phi*R*phi')*(phi*(y_n-t)');
end


%% PART 2

data = load('a010_irlsdata.txt','-ASCII');
X = data(:,1:2); C = data(:,3);

%%
mask = (C==1);
figure(2);
hold on;
plot(X(mask,1),X(mask,2),'r.')
plot(X(~mask,1),X(~mask,2),'b.')
hold off;

%%
% M = mean(X);
% for i = 1:1000
%     X(i,:) = X(i,:) - M;
% end
%%
cl=X(:,1).*X(:,2);
figure(3);
hold on;
scatter3(X(mask,1),X(mask,2),cl(mask),'r')
scatter3(X(~mask,1),X(~mask,2),cl(~mask),'b')
legend('1','0')
xlabel('x_1')
ylabel('x_2')
zlabel('x_3')
hold off;

%% (1, phi_1,phi_2)
Xd = [ones(1,size(X,1));X'];
w = zeros(3,10); 
w(:,1) = [0;0;0];

%initial values 
init_p = arrayfun(@(x1,x2,x3)sigmoid(w(:,1),[x1;x2;x3]),Xd(1,:),Xd(2,:),Xd(3,:));

for i = 2:10
    y_n = arrayfun(@(x1,x2,x3)sigmoid(w(:,i-1),[x1;x2;x3]),Xd(1,:),Xd(2,:),Xd(3,:));
    R = diag(y_n.*(1-y_n));
    w(:,i) = w(:,i-1) - inv(Xd*R*Xd')*(Xd*(y_n-C')');
end

p = arrayfun(@(x1,x2,x3)sigmoid(w(:,end),[x1;x2;x3]),Xd(1,:),Xd(2,:),Xd(3,:));

%%
ce_init = -(sum(C'.*log(init_p)+(1-C').*log(1-init_p))) 
ce = -(sum(C'.*log(p)+(1-C').*log(1-p)))
%%

figure(4);
hold on;
mycolormap = colormap('Jet');
d64 = [0:63]/63; % 
c = interp1(d64, mycolormap,p);
dotsize = 10;
scatter(X(:,1),X(:,2),dotsize,c,'fill');
xlabel('x_1');
ylabel('x_2');
title('Data colored by probability');
colorbar; % what do the colors mean?
hold off;

%% add gaussian basis functions
gaussian = @(x,mu,sigma) exp(-0.5*(x-mu)'*inv(sigma)*(x-mu));
I = [1 0;0 1];
sigma = 0.2*I;
mu_1 = [0 0]';
mu_2 = [1 1]';

phi_1 = @(x) gaussian(x, mu_1, sigma);
phi_2 = @(x) gaussian(x, mu_2, sigma);

PHI = zeros(size(X, 1), 2);
for i=1:size(PHI,1)
    PHI(i,1) = phi_1(X(i,:)');
    PHI(i,2) = phi_2(X(i,:)');
end

figure(41);
hold on;
plot(PHI(mask,1),PHI(mask,2),'r.')
plot(PHI(~mask,1),PHI(~mask,2),'b.')
xlabel('\phi_1')
ylabel('\phi_2')
hold off;

%% PHI' = (1, phi_1,phi_2)
Xd = [ones(1,size(PHI,1));PHI'];
w = zeros(3,10); 
w(:,1) = [0;0;0];

% sigmoid = @(w,x)(1/(1+exp(-(w'*x))));

%initial values 
init_p = arrayfun(@(x1,x2,x3)sigmoid(w(:,1),[x1;x2;x3]),Xd(1,:),Xd(2,:),Xd(3,:));

for i = 2:10
    y_n = arrayfun(@(x1,x2,x3)sigmoid(w(:,i-1),[x1;x2;x3]),Xd(1,:),Xd(2,:),Xd(3,:));
    R = diag(y_n.*(1-y_n));
    w(:,i) = w(:,i-1) - inv(Xd*R*Xd')*(Xd*(y_n-C')');
end

p = arrayfun(@(x1,x2,x3)sigmoid(w(:,end),[x1;x2;x3]),Xd(1,:),Xd(2,:),Xd(3,:));

%%
ce_init = -(sum(C'.*log(init_p)+(1-C').*log(1-init_p))) 
ce = -(sum(C'.*log(p)+(1-C').*log(1-p)))
%%

hold on; figure(5);
mycolormap = colormap('Jet');
d64 = [0:63]/63; % 
c = interp1(d64, mycolormap,p);
dotsize = 10;
scatter(X(:,1),X(:,2),dotsize,c,'fill');
xlabel('x_1');
ylabel('x_2');
title('Data colored by probability');
colorbar; % what do the colors mean?
hold off;