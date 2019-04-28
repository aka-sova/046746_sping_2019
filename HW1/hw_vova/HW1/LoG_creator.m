function [LoG, LoG_double] = LoG_creator(image, levels)
% The function gets an image after imread, and outputs a uint8 LoG tensor
% and a double LoG tensor

    image = double(image); % Converting everything to double allows for an
    %accurate image reconstruction (negative values), without negative
    %values after reconstruction we get a bloom effect

    for i = 1 : 1 : levels

            % Create the sigmas
        
           if i == 1
               sigma = 4;
               sigma_after = 8;
           else
               sigma = 2^i;
               sigma_after = 2^(i+1);
           end

           % Create filters and convert to double
           g_filter = fspecial('gaussian', 4 * sigma, sigma);
           g_filter_after = fspecial('gaussian', 4 * sigma_after, sigma_after);
           
           g_filter = double(g_filter);
           g_filter_after = double(g_filter_after);

           % Create LoG pyramid matrices without downsampling, both double
           % and uint8
           if i == 1
               LoG(:,:,i) = image - imfilter(image, g_filter);
               LoG_double(:,:,i) = double(LoG(:,:,i));
               LoG(:,:,i) = uint8(LoG(:,:,i));
           elseif i == levels
               LoG(:,:,i)= imfilter(image, g_filter);
               LoG_double(:,:,i) = double(LoG(:,:,i));
               LoG(:,:,i) = uint8(LoG(:,:,i));
           else
               LoG(:,:,i) = imfilter(image, g_filter) - imfilter(image, g_filter_after);  
               LoG_double(:,:,i) = double(LoG(:,:,i));
               LoG(:,:,i) = uint8(LoG(:,:,i));
           end

    end

end
