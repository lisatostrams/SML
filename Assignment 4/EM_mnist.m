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
% fid = fopen ('a012_labels.dat', 'r');
% Z = fread(fid, N, 'uint8');
% mu2 = mean(X(Z==2,:));
% mu3 = mean(X(Z==3,:));
% mu4 = mean(X(Z==4,:));
% mu = [mu2;mu3;mu4];

mu = random('unif', 0.25, 0.75, [K,D]);
pi = ones(K,1)/K;

%%
iterations = 40;
[it,gamma] = Bernoulli_EM(X,mu,pi,iterations);

%%
steps = [1:5:20, iterations+1];
rows = length(steps);
%%
figure();
colormap(gray(255));    
for row=1:rows
    mu = squeeze(it(steps(row),:,1:D));
    % map means to grayscale
    mu = uint8(mu * 255);

    for k=1:K
        subplot(rows,K,(row-1)*K+k);
        image(reshape(mu(k,:),[28,28]));
        if(row==1)
            title(sprintf('Class %d', k));   
        end
        if(k==1)
            ylabel(sprintf('Iter %d',steps(row)-1))
            ylabel(sprintf('Iter %d',steps(row)))
        end
        set(gca,'yticklabel',[]);
        set(gca,'xticklabel',[]);
    end
end

%%
fid = fopen ('a012_labels.dat', 'r');
Z = fread(fid, N, 'uint8');
[argvalue, argmax] = max(gamma');
incorrect = 0;

%%
for i = 1:K
    figure; colormap(BW_map);
    hold on;
    Xi = X(argmax==i,:);
    Labelsi = Z(argmax==i,:);
    Label = mode(Labelsi);
    for s=1:min(36,size(Xi,1))
        subplot(6,6,s);
        image(reshape(Xi(s,:),[28,28]));
        set(gca,'yticklabel',[]);
        set(gca,'xticklabel',[]);
    end
    correct = mean(Labelsi==Label);
    Lii = Labelsi(Labelsi~=Label);
    second = mode(Lii); 
    Liii = Lii(Lii~=second);
    third = mode(Liii);
    incorrect = incorrect + sum(Labelsi ~= Label);
    suptitle(sprintf('Cluster %d, most common true label %d occurs %.1f%%. \n Second most common label %d occurs %.1f%%',...
        i,Label,correct*100,second,mean(Labelsi==second)*100))
    hold off;
end
hold off;

fprintf('%.1f %% of the data points is classified incorrectly.\n',incorrect/N*100)

%%
[X2, M] = imread('4.png','png');
X2 = int8(mean(X2,3));
X2 = reshape(X2,[1,784]);
%%
m2 =  mean(X2);
X2(X2 <= m2) = 1;
X2(X2 > m2) = 0;
%%
figure; colormap(BW_map);
imagesc(reshape(X2,[28,28]))
set(gca,'YDir','reverse')
%%
mu = squeeze(it(end,:,1:D));
pi = squeeze(it(end,:,end))';
gamma2 = expectation(X2,mu,pi);