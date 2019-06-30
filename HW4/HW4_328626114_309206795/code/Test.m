
%A_matrix = A_matrix / 255;

image_size = size( image );
A_matrix = reshape( double( image ) , [ image_size( 1 ) * image_size( 2 ) , image_size( 3 ) ] );

[ ~ , Sigma , V ] = svd( A_matrix );



figure(1001)
imshow( uint8( sorted_eimage( : , : , 2 ) ) )

return;