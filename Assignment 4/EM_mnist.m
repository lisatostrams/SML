%% load the data
N = 800; 
D = 28*28;
K = 4;
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

mu = random('unif', 0.25, 0.75, [K,D]);
pi = ones(K,1)/K;

%%
iterations = 40;
[it,gamma] = Bernoulli_EM(X,mu,pi,iterations);

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
        if(k==1)
            ylabel(sprintf('Iter %d',steps(row)))
        end
    end
end

%%
fid = fopen ('a012_labels.dat', 'r');
Z = fread(fid, N, 'uint8');
[argvalue, argmax] = max(gamma');
incorrect = 0;
for i = 1:K
    figure; colormap(BW_map);
    hold on;
    Xi = X(argmax==i,:);
    Labelsi = Z(argmax==i,:);
    Label = mode(Labelsi);
    for s=1:min(36,size(Xi,1))
        subplot(6,6,s);
        image(reshape(Xi(s,:),[28,28]));
    end
    correct = mean(Labelsi==Label);
    Lii = Labelsi(Labelsi~=Label);
    second = mode(Lii); 
    Liii = Lii(Lii~=second);
    third = mode(Liii);
    incorrect = incorrect + sum(Labelsi ~= Label);
    suptitle(sprintf('Cluster %d, most common true label %d occurs %.1f%%. \n second most common label %d occurs %.1f%%',...
        i,Label,correct*100,second,mean(Labelsi==second)*100))
    hold off;
end
hold off;

fprintf('%.1f %% of the data points is classified incorrectly.\n',incorrect/N*100)