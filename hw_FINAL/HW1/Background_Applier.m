function [enhanced_image] = Background_Applier(image, background, mask)
% This function gets an image, a background, and a mask from imread, it
% outputs an enhanced image with changed background

% Converting to double, to bypass any size restrictions
image = double( image );
mask = double( mask );
background = double( background );

% Computation
enhanced_image = ( 255 - mask ) / 255 .* background + mask / 255 .* image;

% Converting back to uint8
enhanced_image = uint8( enhanced_image );

end

