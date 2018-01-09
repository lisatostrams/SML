function f = kernel(theta, x, x_prime)
    theta_0 = theta(1);
    theta_1 = theta(2);
    theta_2 = theta(3);
    theta_3 = theta(4);
    
    norm_2 = (x-x_prime)'*(x-x_prime);
    
    f = theta_0 * exp(-(theta_1/2)*norm_2) + theta_2 + theta_3 * (x'*x_prime);  
end