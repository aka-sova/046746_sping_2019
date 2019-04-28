function [gaussian_pyramid] = make_gaussian_pyramid (image,layers_num)

% the function creates a gaussian pyramid,
% and saves it as a struct


% 1. create the filter of the gaussian


for layer=1:layers_num    
    
    
    % last layer
    sigma = 2^layer;
    gauss_filter = fspecial('gaussian',sigma*5,sigma); 
    gaussian_pyramid{layer} = imfilter(image,gauss_filter);
end

end

