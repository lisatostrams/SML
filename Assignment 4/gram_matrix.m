function f = gram_matrix(theta, X, k)
    N = length(X);
    K_ = zeros(N,N);
    %% populate the gram matrix. Only compute half the matrix, then mirror it
    for n=1:N
        for m=n:N
            K_(n,m) = k(theta, X(n), X(m));
        end
    end
    % mirror the matrix (copy back the diagonal)
    K_ = K_'+K_;
    K_(1:N+1:end) = diag(K_)/2;
    
    f = K_;
end