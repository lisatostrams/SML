%Exercise 1
close all; clear all; 
%% 1

%function
f_x = @(x)(1+sin(6*(x-2))); 

% Training set
D_10_input = linspace(0,1,10);
noiseD = random('norm', 0, 0.3,[1,10]);
%noisy observations to train on
D_10 = f_x(D_10_input)+noiseD;

%Test set
T_input = linspace(0,1,100);
noiseT = random('norm', 0,0.3,[1,100]);
%noisy observations to test on
T = f_x(T_input)+noiseT;

figure; hold on; 
plot(linspace(0,1,100),f_x(linspace(0,1,100)));
plot(D_10_input, D_10,'ro')
legend('Function','Noisy observations')
hold off;
%% 2

%% 3
figure;
Erms_d10 = zeros(10,1);
Erms_T = zeros(10,1);
for m = 0:9
   %calculate weights from training set
   w = PolCurFit([D_10_input;D_10],m);
   
   %compute E root mean square
   Erms_d10(m+1) = E(w,D_10_input,D_10); 
   Erms_T(m+1) = E(w,T_input,T); 
   
   % to plot the polynomial
   t_hat = zeros(1,100);
   x_hat = linspace(0,1,100);
   for j = 1:100
       t = 0;
       for i = 1:m+1
           t = t +(w(i)*(x_hat(j)^(i-1)));
       end
       t_hat(j) =t;
   end
  
   subplot(3,4,m+1); hold on;
   plot(linspace(0,1,100),f_x(linspace(0,1,100)));
   plot(D_10_input, D_10,'ro')
   plot(linspace(0,1,100),t_hat, 'k--')
   title(sprintf('%d-order polynomial estimate',m))
   hold off; 
    
end
legend1 = legend('Function','Noisy observations','Estimate');
set(legend1,'Position',[0.65 0.15 0.1 0.1]);
%%
figure; hold on; 
plot(0:9, Erms_d10,'r--')
plot(0:9, Erms_T, 'k-')
legend('Erms Training set', 'Erms Test set' );
title('Root mean square errors (D10)'); 
hold off; 

%% Repeat for D_40

% Training set
D_40_input = linspace(0,1,40);
noiseD = random('norm', 0, 0.3,[1,40]);
%noisy observations to train on
D_40 = f_x(D_40_input)+noiseD;

figure; hold on; 
plot(linspace(0,1,100),f_x(linspace(0,1,100)));
plot(D_40_input, D_40,'ro')
legend('Function','Noisy observations')
hold off;

%%

figure;
Erms_d40 = zeros(10,1);
Erms_T40 = zeros(10,1);
for m = 0:9
   %calculate weights from training set
   w = PolCurFit([D_40_input;D_40],m);
   
   %compute E root mean square
   Erms_d40(m+1) = E(w,D_40_input,D_40); 
   Erms_T40(m+1) = E(w,T_input,T); 
   
   % to plot the polynomial
   t_hat = zeros(1,100);
   x_hat = linspace(0,1,100);
   for j = 1:100
       t = 0;
       for i = 1:m+1
           t = t +(w(i)*(x_hat(j)^(i-1)));
       end
       t_hat(j) =t;
   end
  
   subplot(3,4,m+1); hold on;
   plot(linspace(0,1,100),f_x(linspace(0,1,100)));
   plot(D_40_input, D_40,'ro')
   plot(linspace(0,1,100),t_hat, 'k--')
   title(sprintf('%d-order polynomial estimate',m))
   hold off; 
    
end
legend1 = legend('Function','Noisy observations','Estimate');
set(legend1,'Position',[0.65 0.15 0.1 0.1]);
%%
figure; hold on; 
plot(0:9, Erms_d40,'r--')
plot(0:9, Erms_T40, 'k-')
legend('Erms Training set', 'Erms Test set' );
title('Root mean square errors (D40)'); 
hold off; 

%% repeat for D10 with lambda > 0

figure;
Erms_d10 = zeros(10,1);
Erms_T = zeros(10,1);
for m = 0:9
   %calculate weights from training set
   w = PolCurFit([D_10_input;D_10],m, 0.0001);
   
   %compute E root mean square
   Erms_d10(m+1) = E(w,D_10_input,D_10); 
   Erms_T(m+1) = E(w,T_input,T); 
   
   % to plot the polynomial
   t_hat = zeros(1,100);
   x_hat = linspace(0,1,100);
   for j = 1:100
       t = 0;
       for i = 1:m+1
           t = t +(w(i)*(x_hat(j)^(i-1)));
       end
       t_hat(j) =t;
   end
  
   subplot(3,4,m+1); hold on;
   plot(linspace(0,1,100),f_x(linspace(0,1,100)));
   plot(D_10_input, D_10,'ro')
   plot(linspace(0,1,100),t_hat, 'k--')
   title(sprintf('%d-order polynomial estimate',m))
   hold off; 
    
end
legend1 = legend('Function','Noisy observations','Estimate');
set(legend1,'Position',[0.65 0.15 0.1 0.1]);
%%

