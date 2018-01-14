%% load the data
N = 800; 
D = 28*28;
K = 3;
X = uint8(zeros(N,D));
fid = fopen('a012_images.dat', 'r'); 
for row = 1:N 
    X(row,:) = fread(fid, D, 'uint8'); 
end 
status = fclose(fid);

%% show some sample images
% for reproducibility
rng(42);
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

mu = random('unif', 0.5, 0.5, [K,D]);
pi = ones(K,1)/K;

%%
iterations = 40;
it = Bernoulli_EM(X,mu,pi,iterations);

%%
steps = [1:4, iterations];
rows = length(steps);
%%
figure();
colormap(gray(255));    
for row=1:rows
    mu = squeeze(it(steps(row),:,1:D));
    pi = squeeze(it(steps(row),:,end));

    % map means to grayscale
    mu = uint8(mu * 255);

    for k=1:K
        subplot(rows,K,(row-1)*K+k);
        axis('square');
        image(reshape(mu(k,:),[28,28]));
        title(sprintf('Class %d', k));        
    end
end