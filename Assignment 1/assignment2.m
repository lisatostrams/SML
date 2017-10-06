
clear all; close all;
%% 1 

h_x_y = @(x,y)(100.*(y-x.^2).^2 + (1-x).^2); 
x = linspace(-2,2,100);
y = linspace(-1,3,100); 
Z = zeros(100,100);
for i = 1:100
    for j = 1:100
        Z(i,j) = h_x_y(x(i),y(j));
    end
end
figure; hold on; 
surf(x,y,Z)
xlabel('X')
ylabel('Y')
hold off; 
%%
figure; 
[X,Y] = meshgrid(-2:.2:2, -1:.2:3);
Z = h_x_y(X,Y);
contour(X,Y,Z);
hold on;
%%

xn_1 = [-2;-1];
delta = 0.1;
while(delta>0.0000001)
    xn_1 = [xn_1 gradient_descent(xn_1(:,end),0.0001)];
    delta = sum(abs(xn_1(:,end)-xn_1(:,end-1)))/2;
    
end
xn_1(:,end)
%%


plot(xn_1(1,:),xn_1(2,:),'k-o');
hold off

title('a contour plot and a gradient descend path');
xlabel('x')
ylabel('y');
