



%% train the gaussian and laplasian



parrots = imread('kodim23.png');
parrot_gray = rgb2gray(parrots);

num_layers = 6;
gaussian_pyramid = make_gaussian_pyramid(parrot_gray,num_layers);
log_pyramid = make_log_pyramid(parrot_gray,num_layers);


reconstructed_parrots = reconstruct_from_log(log_pyramid);

fig23 = figure();
subplot(2,1,1);
imshow(parrot_gray,[]);
title('Original image');
subplot(2,1,2);
imshow(reconstructed_parrots,[]);
title('Reconstructed image');