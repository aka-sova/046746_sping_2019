clear
close all
clc

addpath('eigenFaces');
run('readYaleFaces.m');
image = zeros( 243 , 320 , 20 );

for i = 1:20 % The image numbering is bad programing; put all images in a single variable
    testFileName = ['image' num2str(i)];
    image( : , : , i ) = double( eval ( testFileName ) );
end

%%
% A - is the training set matrix where each column is a face image
% train_face_id - an array with the id of the faces of the training set.
% image1--image20 are the test set.
% is_face - is an array with 1 for test images that contain a face
% faec_id - is an array with the id of the face in the test set, 
%           0 if no face and -1 if a face not from the train-set.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Your Code Here  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

average_image = zeros( 243 , 320 );
for i = 1 : 1 : 20
    
    average_image = average_image + double( image( : , : , i ) ) / 20;
    
end

figure(101)
imshow( uint8( average_image ) );

% Subtract mean image (1.b)

subtracted_image = zeros( 243 , 320 );
for i = 1 : 1 : 20
    
    subtracted_image( : , : , i ) = ( image( : , : , i ) - average_image );
    
end

figure(102)
imshow( uint8( subtracted_image( : , : , 3 ) + 126 ) );

% Compute eigenvectors and report first 5 eigen-faces (2)

eimage = eigenimages( subtracted_image / sqrt(20) , 5 );

for i = 1 : 1 : 5
    figure(200+i)
    imshow( uint8( eimage( : , : , i ) ) + 126 )
end

% Display and compute the representation error for the training images (3)

W_size = 20;
Dr_RMSE = zeros( W_size , 1 );
RMSE = zeros( W_size , 1 );

eimage_large = eigenimages( subtracted_image / sqrt(20) , W_size );
W_matrix = reshape( eimage_large , 243 * 320 , W_size );

for i = 1 : 1 : 20
    
    y_rep = W_matrix' * reshape( subtracted_image( : , : , i ) , 243 * 320 , 1 );
    y_rep = y_rep / (sum(y_rep));
    reconstructed = reshape( average_image , [ 320 * 243 , 1] ) + W_matrix * y_rep;
%     reconstructed = ( reconstructed ) /...
%         ( max( reconstructed(:) ) ) * 255; 
    reconstructed = ( reconstructed - min( reconstructed(:) ) ) /...
        ( max( reconstructed(:) ) - min( reconstructed(:) ) ) * 255; 
    reconstructed = reshape( reconstructed , [ 243 , 320 ] );
    figure(301)
    imshow( uint8( reconstructed )  )
    figure(302)
    imshow( uint8( image( : , : , i ) ) )
    figure(303)
    imshow( uint8( reshape( W_matrix * y_rep , [ 243 , 320 ] ) ) )
    
end

% Compute the representation error for the test images. Classify the test images and report error rate (4)


return;