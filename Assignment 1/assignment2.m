
clear all; close all;
%% 1 

h_x_y = @(x,y)(100.*(y-x.^2).^2 + (1-x).^2); 
x = linspace(-2,2,100);
y = linspace(-1,1,100); 
Z = zeros(100,100);
for i = 1:100
    for j = 1:100
        Z(i,j) = h_x_y(x(i),y(j));
    end
end
figure; hold on; 
surf(x,y,Z)

hold off; 