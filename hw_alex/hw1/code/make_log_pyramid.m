function [log_pyramid] = make_log_pyramid (image,layers_num)

% the function creates a gaussian pyramid,
% and saves it as a struct


% 1. create the filter of the gaussian


for layer=1:layers_num    
    
    
    if (layer==1)
        % first layer
        sigma = 2^2;
        gauss_filter = fspecial('gaussian',sigma*5,sigma); 
        log_pyramid{layer} = image-imfilter(image,gauss_filter);
        
    elseif (layer == layers_num)
        
        % last layer
        sigma = 2^layer;
        gauss_filter = fspecial('gaussian',sigma*5,sigma); 
        log_pyramid{layer} = imfilter(image,gauss_filter);
        
    else 
        % intermediate layers
        sigma_1 = 2^layer;
        gauss_filter_1 = fspecial('gaussian',sigma_1*5,sigma_1); 
        
        sigma_2 = 2^(layer+1);
        gauss_filter_2 = fspecial('gaussian',sigma_2*5,sigma_2); 
        
        log_pyramid{layer} = imfilter(image,gauss_filter_1)-imfilter(image,gauss_filter_2);
    end
end

end

