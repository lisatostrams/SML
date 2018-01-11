%% load the data
N = 800; 
D = 28*28;
K = 3;
X = uint8(zeros(N,D));
fid = fopen('a012_images.dat', 'r'); 
for step = 1:N 
    X(step,:) = fread(fid, D, 'uint8'); 
end 
status = fclose(fid);

%% show some sample images
% for reproducibility
% rng(42);
ix = randi(N,9,1);
samples = X(ix,:);

BW_map=[1,1,1; 0,0,0]; colormap(BW_map);
hold on;
for s=1:size(samples,1)
    subplot(3,3,s);
    image(reshape(X(s,:),[28,28]));
end
hold off;

%% run the algorithm, try different initializations
for i=1:3
    mu = random('unif', 0.25, 0.75, [K,D]);
    pi = ones(K,1)/K;

    it = Bernoulli_EM(X,mu,pi,40);

    %% display the result
    figure();
    colormap(gray(255));    
    for step=1:4
        mu = squeeze(it(step,:,1:D));
        pi = squeeze(it(step,:,end));

        % map means to grayscale
        mu = uint8(mu * 255);
        
        for k=1:K
            subplot(4,K,(step-1)*K+k);
            image(reshape(mu(k,:),[28,28]));
            title(sprintf('Class: %d', k));        
        end
        
    end
end