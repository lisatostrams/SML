function a007(savePlots)
% function a007(savePlots)
%
% if savePlots == 1, writes EPS files containing plots to current directory
% (potentially overwriting existing files!)

% == PART 1 ==
mu    = [1; 0; 1; 2];
Sigma = [ 0.14 , -0.3  , 0.0 ,  0.2 ;
         -0.3  ,  1.16 , 0.2 , -0.8 ;
          0.0  ,  0.2  , 1.0 ,  1.0 ;
          0.2  , -0.8  , 1.0 ,  2.0 ];
Lambda = inv(Sigma);    % precision matrix

% 1: p(x1, x2| x3=x4=0) = Gauss(mu_p, Sigma_p)
mu_a = mu(1:2);
mu_b = mu(3:4);
Laa  = Lambda([1,2],[1,2]);
Lab  = Lambda([1,2],[3,4]); 
% calculate mean and covariance
Sigma_p = inv(Laa);                                 % = [0.1, -0.1; -0.1, 0.12]
mu_p    = mu_a - inv(Laa)*Lab*([0;0] - mu_b);       % = [0.8; 0.8]

% 2: Random number generation (2D-multivariate Gauss)
randn('state', 10); % set random number generator seed (for reproducibility)
mu_t = mvnrnd(mu_p, Sigma_p, 1);   % this is the 'true' value
% plot this prior distribution + selected value 
mu_t100 = mvnrnd(mu_p, Sigma_p, 100);   % note: different nr.of cases give 
figure(1)                               %       different rand for 2nd.col
set(1,'Name','prior distribution');
hold on
plot(mu_t100(:,1), mu_t100(:,2), 'b.');
plot(mu_t(1,1), mu_t(1,2), '+', 'LineWidth', 3,'MarkerEdgeColor', 'r', 'MarkerSize', 20);
xlabel('\mu_1');
ylabel('\mu_2');
if savePlots == 1
  print('-depsc2','a007_part1_2.eps');
end

% 3: Probdensity plot
% NB: mu_p = [0.8; 0.8]; Sigma_p = [0.1, -0.1; -0.1, 0.12];
x1 = -0.5:.04:2; x2 = -0.5:.04:2;
[X1,X2] = meshgrid(x1,x2);
F = mvnpdf([X1(:) X2(:)],mu_p',Sigma_p);
F = reshape(F,length(x2),length(x1));
figure(2);
set(2,'Name','pdf');
surf(x1,x2,F);
xlabel('x_1');
ylabel('x_2');
zlabel('p(x_1,x_2)');
if savePlots == 1
  print('-depsc2','a007_part1_3.eps');
end

% == PART 2 ==
% 1: select datapoints from 'true' distribution and save to file
randn('state', 123); % set random number generator seed (for reproducibility)
Sigma_t = [2.0 , 0.8 ;
           0.8 , 4.0 ];
N = 1000; % number of data points to generate
dat = mvnrnd(mu_t, Sigma_t, N);
% save to a file
s = 'a007_data.txt';
save(s, 'dat', '-ascii');
% plot data on priorplot
figure(1)
hold on
plot(dat(:,1), dat(:,2), 'g.');
hold on
plot(mu_t(1,1), mu_t(1,2), '+', 'LineWidth', 3,'MarkerEdgeColor', 'r', 'MarkerSize', 20);
xlabel('x_1');
ylabel('x_2');
if savePlots == 1
  print('-depsc2','a007_part2.eps');
end

% 2: read datapoints and calculate mu_ML and Sigma_ML
X = load(s, '-ascii');
% initialize
N = size(X,1);
mu_ML = zeros(2,1);     % column vector
Sigma_ML = zeros(2);    % 2x2 matrix
% calculate mean
mu_ML = sum(X)'/N;
% loop over data
for i = 1:N
  % add to cov.matrix
  Sigma_ML = Sigma_ML + (X(i,:)'-mu_ML)*(X(i,:)'- mu_ML)';
end;
% and take average
Sigma_ML = Sigma_ML / N;
% divide by N-1 instead of N to account for bias
Sigma_unbiased = Sigma_ML * N / (N - 1);
% vectorized version (in matrix form):
% mu_ML = mean(X);
% X = X - ones(N,1)*mu_ML;
% Sigma_ML = X'*X/N;
% Sigma_unbiased = X'*X/(N-1);

% == PART 3 ==
% 1: sequential learning of mu_ML and mu_MAP
% (re)initialize
X = load(s, '-ascii');
mu_NML    = zeros(2,N);     % 2xN matrix of N mu_ML estimates
mu_NMAP   = zeros(2,N);     % 2xN matrix of N (Bayesian) mu_MAP estimates
% 
mu_ML     = zeros(2,1);     % (previous) estimate of mu_ML
mu_MAP    = mu_p;           % start with prior!
Sigma_MAP = Sigma_p;        % 2x2 matrix for posterior covariance estimate

% loop over datapoints ... one by one
for i = 1:N
  % get the next point
  x = X(i,:)';
  % first calculate the new ML estimate
  mu_ML = mu_ML + (x - mu_ML)/i;
  % store ML-value in array
  mu_NML(:,i) = mu_ML;
  % now calculate the new MAP value (using eq.2.)
  % p(x)   = N(x|mu_p,Sigma_p)
  % p(y|x) = N(y|x, Sigma_t)
  % p(x|y) = N(x|S{inv(Sigma_t)*y + inv(Sigma_p)*mu_p},
  %              inv(inv(Sigma_p)+inv(Sigma_t))  (=S)
  % ergo: calculate new S, (=> new Sigma_p)
  S = inv(inv(Sigma_MAP)+inv(Sigma_t));
  mu_MAP = S*(inv(Sigma_t)*x + inv(Sigma_MAP)*mu_MAP);
  Sigma_MAP = S;
  % store MAP-value in array
  mu_NMAP(:,i) = mu_MAP;
  % ... next datapoint
end;

% 2: plot estimates (both components!) as a function of the nr.of datapoints
figure(3)
set(3,'Name','ML-estimate')
hold on
plot(mu_NML(1,:), mu_NML(2,:));
plot(mu_t(1,1), mu_t(1,2), '+', 'LineWidth', 3,'MarkerEdgeColor', 'r', 'MarkerSize', 20);
xlabel('\mu_1');
ylabel('\mu_2');
if savePlots == 1
  print('-depsc2','a007_part3_2_ML.eps');
end

figure(4)
set(4,'Name','MAP-estimate')
hold on
plot(mu_NMAP(1,:), mu_NMAP(2,:));
plot(mu_t(1,1), mu_t(1,2), '+', 'LineWidth', 3,'MarkerEdgeColor', 'r', 'MarkerSize', 20);
xlabel('\mu_1');
ylabel('\mu_2');
if savePlots == 1
  print('-depsc2','a007_part3_2_MAP.eps');
end

% ML vs. MAP convergence as a function of the nr.of datapoints
figure(5)
set(5,'Name','1D-lineplot, ML vs. MAP convergence')
hold on
% component t1
plot(mu_NML(1,:),'-b');         % 1D-lineplot ML                (blue)
plot(mu_NMAP(1,:),'-g');        % 1D-lineplot MAP               (green)
plot(mu_t(1,1)*ones(1,N),'-k');	% 1D-lineplot true value of t1  (black)
% component t2
plot(mu_NML(2,:),':b');         % 1D-lineplot ML                (blue)
plot(mu_NMAP(2,:),':g');        % 1D-lineplot MAP               (green)
plot(mu_t(1,2)*ones(1,N),':k');	% 1D-lineplot true value of t2  (black)
xlabel('N');
ylabel('\mu_1, \mu_2');
if savePlots == 1
  print('-depsc2','a007_part3_2_N_MLMAP.eps');
end

% ML vs. MAP estimate
figure(6)
set(6,'Name','ML vs. MAP -estimate')
hold on
plot(mu_NML(1,:), mu_NML(2,:), '-b');
plot(mu_NMAP(1,:), mu_NMAP(2,:), '-g');     % note similarity with shape of prior
plot(mu_t(1,1), mu_t(1,2), '+', 'LineWidth', 3,'MarkerEdgeColor', 'r', 'MarkerSize', 20);
xlabel('\mu_1');
ylabel('\mu_2');
if savePlots == 1
  print('-depsc2','a007_part3_2_MLMAP.eps');
end

% final posterior distribution
F2 = mvnpdf([X1(:) X2(:)],mu_MAP',Sigma_MAP);
F2 = reshape(F2,length(x2),length(x1));
figure(7)
set(7,'Name','posterior distribution (MAP)')
hold on
surf(x1,x2,F2);
xlabel('\mu_1');
ylabel('\mu_2');
if savePlots == 1
  print('-depsc2','a007_part3_2_MAPposterior.eps');
end

return
