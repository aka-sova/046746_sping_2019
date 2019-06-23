function [ normalized_W_matrix ] = eigenimages( image , r )

% This function creates a sorted array of eigenimages based on the highest
% eigenvalues. It gets an array of grayscale images and outputs an array of
% grayscale eigenimages. The parameter r determines the number of
% eigenimages corresponding to the highest eigenvalues to return.
%

% Flatten each image into a vector of flattened pixel values; concatinate
% all of the vectors into a matrix with number of columns as the number of
% images.
image_size = size( image );
A_matrix = reshape( double( image ) , [ image_size( 1 ) * image_size( 2 ) , image_size( 3 ) ] );

% Create a matrix of A^T * A. In our case it is a 20x20 matrix. Compute all
% eigenvalues and eigenvectors from this matrix.
AtA_matrix = A_matrix' * A_matrix;
[ eigenvectors , eigenvalues ] = eig( AtA_matrix );

% A^T*A* \lambda = X*\lambda
% (A*A^T)*A*\lambda = A*X*\lambda
% So that A*X is the eigenvector
sorted_evec = A_matrix * eigenvectors;

% Sorting by largest eigenvalues
[ ~ , idx ] = sort( diag( eigenvalues ) , 'descend' );
sorted_evec = sorted_evec( : , idx );

% Reshaping the largest eigenvalue image set to be a matrix with the size
% of A
sorted_eimage = reshape( sorted_evec , [ image_size( 1 ) , image_size( 2 ) , image_size( 3 ) ] );
% Extracting 'r' eigenvectors corresponding to the largest eigenvalues
if r >= image_size( 3 )
    r = image_size( 3 );
end
sorted_eimage = sorted_eimage( : , : , 1 : r );
W_matrix = reshape( sorted_eimage , [ image_size( 1 ) * image_size( 2 ) , r ] );
% Normalizing the W matrix such that W^T W = I
WtW_matrix = W_matrix' * W_matrix;
Normalizer = 1 ./ diag( WtW_matrix ).^.5;
normalized_W_matrix = W_matrix * diag( Normalizer );
% Outputing the W matrix, converted to 'r' largest eigenvalue images.
normalized_W_matrix = reshape( normalized_W_matrix , [ image_size( 1 ) , image_size( 2 ) , r ] );


end

