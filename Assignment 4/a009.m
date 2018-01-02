function a009
% function a009
rng(5) % for reproducibility
printflag = 0; % set to a to make pdf exports
% 1: predictive distribution
% constants
alpha = 2;            % precision of prior over weights
beta  = 10;           % precision of data generating process
% data
Xn = [0.4;  0.6];
Tn = [0.05; -0.35];
% calculate averages
N = length(Xn);       % = 2
mu_x  = mean(Xn);     % =  0.5
mu_t = mean(Tn);     % = -0.15   
mu_xt = mean(Xn.*Tn); % = -0.095
mu_xx = mean(Xn.*Xn); % =  0.26
mu_x
mu_t
mu_xt
mu_xx
% calc. SnInv (=inv(Sn)) and Sn
SnInv = alpha*eye(N) + N*beta*[1 , mu_x; mu_x ,  mu_xx];  % = [22, 10; 10, 7.2]
Sn = inv(SnInv)    % =~ [0.1233 , -0.1712; -0.1712 , 0.3767]

% calculate Mn
%mx = Mn(1) + Mn(2)*x;         % mx = -0.0445 - 0.2021*x 
Mn = N*beta*Sn*[mu_t; mu_xt]   % =~ [-0.0445 ; -0.2021]

% compute variance s2(x) (quadratic) s2(x) = 0.2233 - 0.3425*x + 0.3767*x^2
%s2x = 1/beta + Sn(1,1) + (Sn(1,2)+Sn(2,1))*x + Sn(2,2)*x.*x;
sigma2fncoefss = [1/beta + Sn(1,1),(Sn(1,2)+Sn(2,1)),Sn(2,2)]


% 2: plot predictive distribution
% choose x: say 21 points over the interval x = [0, 1]
x = [0:0.05:1];
% compute m(x) (linear, so form m(x) = a + b*x)
mx = Mn(1) + Mn(2)*x;         % mx = -0.0445 - 0.2021*x 

% compute variance s2(x) (quadratic) s2(x) = 0.2233 - 0.3425*x + 0.3767*x^2
s2x = 1/beta + Sn(1,1) + (Sn(1,2)+Sn(2,1))*x + Sn(2,2)*x.*x;
% standard deviation is sqrt(variance)
sx = sqrt(s2x);

figure(11)
set(11,'Name','Predictive distribution')
axis([0 1 -1.5 1.5]);

xlabel('x');
ylabel('y');
hold on
% first the m(x)+/-s(x) areas (no line)
area(x,(mx+sx), 'FaceColor', [1.0, 0.8, 0.8], 'BaseValue',-1.5);  % pinkish
area(x,(mx-sx), 'FaceColor', [1.0, 1.0, 1.0], 'BaseValue',-1.5);  % white
% the lines for the predictive mean m(x) and variance s(x) around it
plot(x,(mx+sx),'r', 'LineWidth',2);     % red
plot(x,mx,'k','LineWidth',2);                         % black
plot(x,(mx-sx),'r', 'LineWidth',2);     % red
% circle the datapoints
plot(Xn,Tn,'o','MarkerEdgeColor','k','LineWidth',2, 'MarkerSize',10);

% unimportant print stuff
ps = get(gcf, 'Position');
ratio = (ps(4)-ps(2)) / (ps(3)-ps(1))
paperWidth = 10;
paperHeight = paperWidth*ratio;
set(gcf, 'paperunits', 'centimeters');
set(gcf, 'papersize', [paperWidth paperHeight]);
set(gcf, 'PaperPosition', [0    0   paperWidth paperHeight]);
if(printflag)
print -dpdf a009_ex2
end;
pause;
% In case (try!):
% * alpha => 0   :=> m(x) = 0.85 - 2*x; (line through points)
% * beta  => Inf :=> m(x) = 0.85 - 2*x; s2(x) = 0; 

% 3: sample five functions
ncases = 5;
% the posterior distribution over w is a multivariate Gaussian
W = mvnrnd(Mn', Sn, ncases);
Phi_x = [ones(1,length(x));x];
Y = W*Phi_x;    % compute y = w(1) + w(2)*x for five functions at all points x
% plot the lines in the same figure (11)
plot(x,Y,'b','LineWidth',1.5);            % blue
if(printflag)
print -dpdf a009_ex3
end;
return
