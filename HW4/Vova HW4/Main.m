clear
close all
clc

addpath(genpath('eigenFaces'));
run('readYaleFaces.m');
test_image = zeros( 243 , 320 , 20 );
image = reshape( A , [ 243 , 320 , 150 ]);

for i = 1:20 % The image numbering is bad programing; put all images in a single variable
    testFileName = ['image' num2str(i)];
    test_image( : , : , i ) = double( eval ( testFileName ) );
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
for i = 1 : 1 : 150
    
    average_image = average_image + double( image( : , : , i ) ) / 150;
    
end

figure(101)
imshow( uint8( average_image ) );

% Subtract mean image (1.b)

subtracted_image = zeros( 243 , 320 );
for i = 1 : 1 : 150
    
    subtracted_image( : , : , i ) = ( image( : , : , i ) - average_image );
    
end

% Compute eigenvectors and report first 5 eigen-faces (2)

eimage = eigenimages( subtracted_image / sqrt(150) , 5 );

for i = 1 : 1 : 5
    figure(200+i)
    step_image = eimage( : , : , i );
    normalized_image = step_image / max( step_image(:) ) * 126; 
    imshow( uint8( normalized_image + 126 ) )
end

% Display and compute the representation error for the training images (3)

W_size = 25;
Dr_RMSE_squared = zeros( 150 , 1 );
RMSE_squared = zeros( 150 , 1 );

eimage_large = eigenimages( subtracted_image / sqrt(150) , W_size );
W_matrix = reshape( eimage_large , 243 * 320 , W_size );

for i = 1 : 1 : 150
    
    y_rep = W_matrix' * reshape( subtracted_image( : , : , i ) , 243 * 320 , 1 );
    reconstructed = reshape( average_image , [ 320 * 243 , 1] ) + W_matrix * y_rep;
    reconstructed = reshape( reconstructed , [ 243 , 320 ] );
%     figure(301)
%     imshow( uint8( reconstructed )  )
%     figure(302)
%     imshow( uint8( image( : , : , i ) ) )
%     figure(303)
%     imshow( uint8( reshape( W_matrix * y_rep , [ 243 , 320 ] ) ) )
    
    normalized_training_image = image( : , : , i ) / ( max( image( : , : , i ) ) - min( image( : , : , i ) ) );
    normalized_reconstruction = reconstructed / ( max( reconstructed ) - min( reconstructed ) );
    
    Dr_RMSE_squared( i ) = sum(sum( ( normalized_reconstruction...
        - normalized_training_image ).^2 )) / ( 243 * 320 * 150 );
    
    RMSE_squared( i ) = sum(sum( ( reconstructed / 255 ...
        - image( : , : , i ) / 255 ).^2 )) / ( 243 * 320 * 150 );
    
end

Dr_RMSE = Dr_RMSE_squared.^.5;
RMSE = RMSE_squared.^.5;

figure(310)
hold on
plot(RMSE, 'r')
plot(Dr_RMSE)
xlabel( 'Image number' )
ylabel( 'Error' )
legend( 'RMSE' , 'Dynamic range RMSE')

% Compute the representation error for the test images. Classify the test images and report error rate (4)

subtracted_test_image = zeros( 243 , 320 , 20 );
for i = 1 : 1 : 20
    subtracted_test_image( : , : , i ) = test_image( : , : , i ) - average_image;
end

for i = 1 : 1 : 20
    
    y_rep = W_matrix' * reshape( subtracted_test_image( : , : , i ) , 243 * 320 , 1 );
    reconstructed = reshape( average_image , [ 320 * 243 , 1] ) + W_matrix * y_rep;
    reconstructed = reshape( reconstructed , [ 243 , 320 ] );
    figure(401)
    imshow( uint8( reconstructed )  )
    figure(402)
    imshow( uint8( test_image( : , : , i ) ) )
    figure(403)
    imshow( uint8( reshape( W_matrix * y_rep , [ 243 , 320 ] ) ) )
    
    normalized_test_image = test_image( : , : , i ) / ( max( test_image( : , : , i ) ) - min( test_image( : , : , i ) ) );
    normalized_reconstruction = reconstructed / ( max( reconstructed ) - min( reconstructed ) );
    
    Dr_RMSE_squared( i ) = sum(sum( ( normalized_reconstruction...
        - normalized_test_image ).^2 )) / ( 243 * 320 * 150 );
    
    RMSE_squared( i ) = sum(sum( ( reconstructed / 255 ...
        - test_image( : , : , i ) / 255 ).^2 )) / ( 243 * 320 * 150 );
    
end

Dr_RMSE = Dr_RMSE_squared.^.5;
RMSE = RMSE_squared.^.5;

figure(410)
hold on
plot(RMSE, 'r')
plot(Dr_RMSE)
xlabel( 'Image number' )
ylabel( 'Error' )
legend( 'RMSE' , 'Dynamic range RMSE')

return;