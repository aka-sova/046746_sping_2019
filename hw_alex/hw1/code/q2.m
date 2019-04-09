



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



% very strange - the results are not the same.
% try - even adding and substracting blurred image will give different
% results


% for example, this one will NOT result in the original image:
sigma = 2^2;
gauss_filter = fspecial('gaussian',sigma*5,sigma); 
test_image = parrot_gray-imfilter(parrot_gray,gauss_filter)+imfilter(parrot_gray,gauss_filter);


% but if we allow the values to become negative, it will reconstruct it
% correctly.
log_image = int16(parrot_gray)-int16(imfilter(parrot_gray,gauss_filter));
final_image = uint8(log_image+int16(imfilter(parrot_gray,gauss_filter)));


% I sent an email regarding the type of structure in LoG. 
% How did you implement it?




