function [image_out] = stitch( image_1 , image_2 )
% Stitching function; Gets as input 2 images, background image 1 and patch
% image 2. The function outputs image_out RGB image with double variable
% type.

image_1 = double( image_1 );
image_out = image_1;
image_size = size( image_2 );

if length( image_size ) < 3
    image_size( 3 ) = 1;
end

% Filters out all NaN points or black points (up to a certain threshold)
for i = 1 : 1 : image_size( 1 )
    for j = 1 : 1 : image_size( 2 )
        for k = 1 : 1 : image_size( 3 )
            
            if ~isnan(image_2( i , j , k ) ) && sum(image_2( i , j , : ).^2) > 1 
                
                image_out( i , j , k ) = image_2( i , j , k );
                           
            end
            
        end
    end
end

end

