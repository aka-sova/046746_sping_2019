image_1 = Stop_3;
image_2 = affine_transformation( Stop_1 , hom_3_max );

%% Algorithm

image_1 = double( image_1 );
image_out = image_1;
image_size = size( image_2 );

% Filters out all NaN points or black points (up to a certain threshold)
for i = 1 : 1 : image_size( 1 )
    for j = 1 : 1 : image_size( 2 )
        for k = 1 : 1 : image_size( 3 )
            
            if ~isnan(image_2( i , j , k ) ) && sum(image_2( i , j , : ).^2) > 1 
                
                image_out( i ,j , k ) = image_2( i , j , k );
                           
            end
            
        end
    end
end

imshow( uint8( image_out ) );