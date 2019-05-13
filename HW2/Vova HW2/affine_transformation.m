function [ interpolated_image ] = affine_transformation( image , affine_trans )
% This function gets an image and an affine transformation matrix, it
% outputs an image in the double format after transformation by the affine
% matrix.

% Converting to double and extracting size
image = double( image );
image_size = size( image );

if length( image_size ) < 3
    image_size( 3 ) = 1;
end

[ grid_x, grid_y ] = meshgrid( 1 : image_size( 1 ) , 1 : image_size( 2 ) );

% Project each coordinate to a projected one
for i = 1 : 1 : image_size( 1 )
    for j = 1 : 1 : image_size( 2 )
        
        projected_vector = affine_trans * [ grid_x( i , j ) ; grid_y( i , j ) ; 1 ];
        grid_x_new( i , j ) = projected_vector( 1 );
        grid_y_new( i , j ) = projected_vector( 2 );
        
    end
end

% Round each coordinate to the closest integer
grid_x_new_rounded = round( grid_x_new );
grid_y_new_rounded = round( grid_y_new );
affine_inverse = affine_trans ^ (-1);

x_max_coordinate = max( max( grid_x_new_rounded ) );
y_max_coordinate = max( max( grid_y_new_rounded ) );

[ grid_x_out , grid_y_out ] = meshgrid( 1 : x_max_coordinate , 1 : y_max_coordinate );

% Inverse transformation from the rounding action for interp2 to use
for i = 1 : 1 : y_max_coordinate
    for j = 1 : 1 : x_max_coordinate
        
        projected_vector = affine_inverse * [ grid_x_out( i , j ) ; grid_y_out( i , j ) ; 1 ];
        grid_x_back( i , j ) = projected_vector( 1 );
        grid_y_back( i , j ) = projected_vector( 2 );
        
    end
end

interpolated_image = NaN( y_max_coordinate , x_max_coordinate , image_size( 3 ) );

% Activating interp2 on a color image
for i = 1 : 1 : image_size( 3 )

    interpolated_image( : , : , i ) = interp2( grid_x , grid_y , image( : , : , i ) ,...
        grid_x_back , grid_y_back );

end


end

