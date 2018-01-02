function a015
% function a015
  disp 'a015 - part 2.(e) of the exercise';
  a015_2
% print('-depsc2','../fig/a015_2.eps');

  disp 'a015 - part 3.(c) of the exercise';
  a015_3
% print('-depsc2','../fig/a015_3.eps');
return


function a015_2
% function a015_2
% corresponding to part 2.(e) of the exercise
  a1 = 1;
  a2 = 2;
  maxiter = 1000;
  tol = 1e-9;

  % useful for contour plot later on
  [X,Y] = meshgrid(-1:.2:3, 0:.2:4);

  % plot for various values of lambda1,lambda2
  lambdas = [1,1;
             1,10];
  figure;
  for lambdaindex = 1:2
    lambda1 = lambdas(lambdaindex,1);
    lambda2 = lambdas(lambdaindex,2);

    % calculate optimal learning rate
    eta_opt = 2./(lambda1 + lambda2);

    % prepare for plotting contours of g
    % the function to be minimized
    g = @(x,y) (lambda1 / 2) * (x - a1).^2 + (lambda2 / 2) * (y - a2).^2;
    Z = g(X,Y);

    % plot for various values of eta
    eta = 0.2 * eta_opt;
    etas = [0.2 * eta_opt, eta_opt, 1.5 * eta_opt];
    etalabels = {'small','optimal','large'};
    for etaindex = 1:3
      eta = etas(etaindex);
      etalabel = etalabels{etaindex};
      [xs1,ys1,dists1,iter] = a015_2_graddesc(a1,a2,lambda1,lambda2,eta,a1+1,a2+1,maxiter,tol);
      [xs2,ys2,dists2,iter] = a015_2_graddesc(a1,a2,lambda1,lambda2,eta,a1-0.5,a2+1,maxiter,tol);
      [xs3,ys3,dists3,iter] = a015_2_graddesc(a1,a2,lambda1,lambda2,eta,a1-1,a2-0.5,maxiter,tol);
      [xs4,ys4,dists4,iter] = a015_2_graddesc(a1,a2,lambda1,lambda2,eta,a1+0.5,a2-1,maxiter,tol);
      subplot(2,3,etaindex+(lambdaindex-1)*3);
      contour(X,Y,Z);
      hold on;
      plot(xs1,ys1,'k-o',xs2,ys2,'k-+',xs3,ys3,'k-d',xs4,ys4,'k-x');
      hold off;
      title(sprintf('\\lambda1 = %.0f, \\lambda2 = %.0f, \\eta = %.3f (%s)', lambda1, lambda2, eta, etalabel));
      xlim([-1,3]);
      ylim([0,4]);
    end
  end
return


function a015_3
% function a015_3
% corresponding to part 3.(a) of the exercise

  % define the function h(x,y)
  h = @(x,y) 100 * (y - x.^2).^2 + (1 - x).^2;
  % define x-values
  x = [-2:0.1:2];
  % define y-values
  y = [-1:0.1:3];

  % define the derivative of the function h(x,y)
  dhdxy = @(x) [400 * x(1).^3 - 400 * x(2) .* x(1) + 2 * x(1) - 2, 200 * x(2) - 200 * x(1).^2];

  % plot trajectories for different values of eta
  etas = [1e-4, 1e-3, 3e-3];
  figure;
  for etaindex = 1:3
    eta = etas(etaindex);

    % run gradient descent
    [xs,dists,iter] = grad_descent(dhdxy,[1,-1],eta,10000,1e-9);
    disp(sprintf('eta = %f needs %d iterations to obtain a tolerance of %f\n', eta, iter, dists(end)))

    % plot distance vs. iterations
    subplot(2,3,3+etaindex);
    plot(1:400,log10(dists(1:400)),'k-');
    xlabel('n');
    ylabel('log_{10} ||x_n - x_{n-1}||');

    % plot trajectory
    subplot(2,3,etaindex);
    contour(x,y,h(ones(41,1)*x,y'*ones(1,41)),50);
    hold on;
    plot(xs(1:400,1),xs(1:400,2),'k-o');
    hold off;
    xlim([-2,2]);
    ylim([-1,3]);
    xlabel('x');
    ylabel('y');
    title(sprintf('\\eta = %f',eta));
  end
return


function [xs,ys,dists,iter] = a015_2_graddesc(a1,a2,lambda1,lambda2,eta,x0,y0,maxiter,tol)
% function [xs,ys,dists,iter] = a015_2_graddesc(a1,a2,lambda1,lambda2,eta,x0,y0,maxiter,tol)
%
% This function implements the gradient descent algorithm for the 
% g function defined in the assignment:
% 
%   g(x,y) = (lambda1 / 2) * (x - a1)^2 + (lambda2 / 2) * (y - a2)^2
% 
% INPUT:
%   a1      parameter of g function
%   a2      parameter of g function
%   lambda1 parameter of g function
%   lambda2 parameter of g function
%   eta     gradient descent learning rate
%   x0      initial value for x
%   y0      initial value for y
%   maxiter maximum number of gradient descent iterations
%   tol     tolerance for detecting convergence
%
% OUTPUT:
%   xs      column vector of x coordinates during minimization
%   ys      column vector of y coordinates during minimization
%   dists   column vector of distances to previous point during minimization
%   iter    number of gradient descent iterations done
%

  % the function to be minimized
  g = @(x,y) (lambda1 / 2) * (x - a1)^2 + (lambda2 / 2) * (y - a2)^2;

  % initialization
  x = [x0];
  y = [y0];
  % keep track of everything
  xs = [x];
  ys = [y];
  dists = zeros(0,1);
  
  % gradient descent iterations
  for iter = 1:maxiter
    % calculate next coordinates
    xnew = (1 - eta * lambda1) * x + eta * lambda1 * a1;
    ynew = (1 - eta * lambda2) * y + eta * lambda2 * a2;

    % calculate distance to current coordinates
    dist = sqrt((xnew - x)^2 + (ynew - y)^2);

    % update coordinates
    x = xnew;
    y = ynew;

    % keep track of everything
    xs = [xs; x];
    ys = [ys; y];
    dists = [dists; dist];

    % if converged, exit the loop
    if dist < tol
      break
    end
  end

return


function [xs,dists,iter] = grad_descent(dfdx,x0,eta,maxiter,tol)
% function [xs,dists,iter] = grad_descent(dfdx,x0,eta,maxiter,tol)
%
% This function implements the gradient descent algorithm. The user
% has to specify the derivative of the function.
% 
% INPUT:
%   dfdx    function handle to a function which when called like
%             u = dfdx(x)
%           returns the derivative of f with respect to x evaluated
%           at x, where x can be a vector and u will be a vector of
%           the same size
%   eta     gradient descent learning rate
%   x0      initial value for x
%   maxiter maximum number of gradient descent iterations
%   tol     tolerance for detecting convergence
%
% OUTPUT:
%   xs      matrix of x values during minimization
%           (the n'th row corresponds to the value of x at the n'th iteration)
%   dists   column vector of distances to previous point during minimization
%   iter    number of gradient descent iterations done

  % initialization
  x = x0;
  % keep track of everything
  xs = [x];
  dists = zeros(0,1);
  
  % gradient descent iterations
  for iter = 1:maxiter
    % calculate next coordinates
    xnew = x - eta * dfdx(x);

    % calculate distance to current coordinates
    dist = norm(xnew - x,2);

    % update coordinates
    x = xnew;

    % keep track of everything
    xs = [xs; x];
    dists = [dists; dist];

    % if converged, exit the loop
    if dist < tol
      break
    end
  end

return
