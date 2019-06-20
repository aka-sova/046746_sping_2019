function [ normalized_W_matrix ] = eigenimages( image , r )

% This function creates a sorted array of eigenimages based on the highest
% eigenvalues. It gets an array of grayscale images and outputs an array of
% grayscale eigenimages. The parameter r determines the number of
% eigenimages corresponding to the highest eigenvalues to return.
%

image_size = size( image );
A_matrix = reshape( double( image ) , [ image_size( 1 ) * image_size( 2 ) , image_size( 3 ) ] );

AtA_matrix = A_matrix' * A_matrix;
[ eigenvectors , eigenvalues ] = eig( AtA_matrix );

sorted_evec = A_matrix * eigenvectors;

[ ~ , idx ] = sort( diag( eigenvalues ) , 'descend' );
sorted_evec = sorted_evec( : , idx );

sorted_eimage = reshape( sorted_evec , [ image_size( 1 ) , image_size( 2 ) , image_size( 3 ) ] );
if r >= image_size( 3 )
    r = image_size( 3 );
end
sorted_eimage = sorted_eimage( : , : , 1 : r );
W_matrix = reshape( sorted_eimage , [ image_size( 1 ) * image_size( 2 ) , r ] );
WtW_matrix = W_matrix' * W_matrix;
Normalizer = 1 ./ diag( WtW_matrix ).^.5;
normalized_W_matrix = W_matrix * diag( Normalizer );
normalized_W_matrix = reshape( normalized_W_matrix , [ image_size( 1 ) , image_size( 2 ) , r ] );
% for i = 1 : 1 : r
%     part_sorted_eimage = sorted_eimage( : , : , i );
%     sorted_eimage( : , : , i ) = ( sorted_eimage( : , : , i ) -...
%         min( part_sorted_eimage(:) ) ) / ( max( part_sorted_eimage(:) ) - min( part_sorted_eimage(:) ) );
% end

end

