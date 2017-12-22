function xn_1 = gradient_descent(xn, eta)
grad_x = 400.*(xn(1,:).^3)+ 2.*xn(1,:) - 400.*(xn(1,:).*xn(2,:)) -2;
grad_y = 200.*xn(2,:) - 200.*xn(1,:); 
xn_1 = [xn(1,:) - eta.*(grad_x); xn(2,:) - eta.*(grad_y)];
end