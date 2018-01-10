%% load the data
N = 800; 
D = 28*28; 
X = uint8(zeros(N,D)); 
fid = fopen('a012_images.dat', 'r'); 
for i = 1:N 
    X(i,:) = fread(fid, [D], 'uint8'); 
end 
status = fclose(fid);

%% show some sample images
% for reproducibility
rng(42);
ix = randi(N,9,1);
samples = X(ix,:);

BW_map=[1,1,1; 0,0,0]; colormap(BW_map);
hold on;
for i=1:size(samples,1)
    subplot(3,3,i);
    image(reshape(X(i,:),[28,28]));
end
hold off;