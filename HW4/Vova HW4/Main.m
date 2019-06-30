clear
close all
clc

mkdir('imgs');

addpath(genpath('eigenFaces'));
run('readYaleFaces.m');
test_image = zeros( 243 , 320 , 20 );

for i = 1:20 % The image numbering is bad programing; put all images in a single variable
    testFileName = ['image' num2str(i)];
    test_image( : , : , i ) = double( eval ( testFileName ) );
end

clearvars image*

image = reshape( A , [ 243 , 320 , 150 ]);


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

fig = figure(101);
imshow( uint8( average_image ) );
saveas(fig, './imgs/average_face.png', 'png');

% Subtract mean image (1.b)

subtracted_image = zeros( 243 , 320 );
for i = 1 : 1 : 150
    
    subtracted_image( : , : , i ) = ( image( : , : , i ) - average_image );
    
end

%% Compute eigenvectors and report first 5 eigen-faces (2)

eimage = eigenimages( subtracted_image / sqrt(150) , 5 );

for i = 1 : 1 : 5
    fig = figure(200+i);
    step_image = eimage( : , : , i );
    normalized_image = step_image / max( step_image(:) ) * 126; 
    imshow( uint8( normalized_image + 126 ) )
    filename = sprintf('./imgs/biggest_ev_%1.0f.png',i);
    saveas(fig, filename, 'png');
end

%% Display and compute the representation error for the training images (3)

W_size = 25;
Dr_RMSE_squared = zeros( 150 , 1 );
RMSE_squared = zeros( 150 , 1 );

eimage_large = eigenimages( subtracted_image / sqrt(150) , W_size );
W_matrix = reshape( eimage_large , 243 * 320 , W_size );
y_rep_train = zeros( W_size , 150 );

reconstruction_presentation = [ 10 , 20 , 30 ];

for i = 1 : 1 : 150 % Compute reconstruction error and present few reconstruction results
    
    y_rep_train( : , i ) = W_matrix' * reshape( subtracted_image( : , : , i ) , 243 * 320 , 1 );
    reconstructed = reshape( average_image , [ 320 * 243 , 1] ) + W_matrix * y_rep_train( : , i );
    reconstructed = reshape( reconstructed , [ 243 , 320 ] );
    
    if ismember( i , reconstruction_presentation )
        figure(3001 + 10 * i)
        imshow( uint8( reconstructed )  )
        figure(3002 + 10 * i)
        imshow( uint8( image( : , : , i ) ) )
    end
    
    normalized_training_image = image( : , : , i ) / ( max( image( : , : , i ) ) - min( image( : , : , i ) ) );
    normalized_reconstruction = reconstructed / ( max( reconstructed ) - min( reconstructed ) );
    
    Dr_RMSE_squared( i ) = sum(sum( ( normalized_reconstruction...
        - normalized_training_image ).^2 )) / ( 243 * 320 * 150 );
    
    RMSE_squared( i ) = sum(sum( ( reconstructed / 255 ...
        - image( : , : , i ) / 255 ).^2 )) / ( 243 * 320 * 150 );
    
end

Dr_RMSE = Dr_RMSE_squared.^.5;
RMSE = RMSE_squared.^.5;

% Compute average error
avg_Dr_RMSE_train = mean(Dr_RMSE);
avg_RMSE_train = mean(RMSE);

% Present error graphs
fig = figure(310);
hold on
plot(RMSE, 'r')
plot(Dr_RMSE)
grid;
xlabel( 'Image number' )
ylabel( 'Error' )
legend( 'RMSE' , 'Dynamic range RMSE')
filename = sprintf('./imgs/rmse_train.png');
saveas(fig, filename, 'png');

%% Compute the representation error for the test images. Classify the test images and report error rate (4)

clearvars Dr_RMSE_squared RMSE_squared Dr_RMSE RMSE reconstruction_presentation

reconstruction_presentation = [ 15 , 1 , 10 ];

subtracted_test_image = zeros( 243 , 320 , 20 );
for i = 1 : 1 : 20
    subtracted_test_image( : , : , i ) = test_image( : , : , i ) - average_image;
end

y_rep_test = zeros( W_size , 20 );

for i = 1 : 1 : 20 % Compute reconstruction error and present few reconstruction results
    
    y_rep_test( : , i ) = W_matrix' * reshape( subtracted_test_image( : , : , i ) , 243 * 320 , 1 );
    reconstructed = reshape( average_image , [ 320 * 243 , 1] ) + W_matrix * y_rep_test( : , i );
    reconstructed = reshape( reconstructed , [ 243 , 320 ] );
    
    if ismember( i , reconstruction_presentation )
        figure(4001 + i * 10)
        imshow( uint8( reconstructed )  )
        figure(4002 + i * 10)
        imshow( uint8( test_image( : , : , i ) ) )
    end
    
    normalized_test_image = test_image( : , : , i ) / ( max( test_image( : , : , i ) ) - min( test_image( : , : , i ) ) );
    normalized_reconstruction = reconstructed / ( max( reconstructed ) - min( reconstructed ) );
    
    Dr_RMSE_squared( i ) = sum(sum( ( normalized_reconstruction...
        - normalized_test_image ).^2 )) / ( 243 * 320 * 150 );
    
    RMSE_squared( i ) = sum(sum( ( reconstructed / 255 ...
        - test_image( : , : , i ) / 255 ).^2 )) / ( 243 * 320 * 150 );
    
end

Dr_RMSE = Dr_RMSE_squared.^.5;
RMSE = RMSE_squared.^.5;

% Compute average error

avg_Dr_RMSE_test = mean(Dr_RMSE);
avg_RMSE_test = mean(RMSE);

% Present error graphs
fig = figure(410);
hold on
plot(RMSE, 'r')
plot(Dr_RMSE)
grid;
xlabel( 'Image number' )
ylabel( 'Error' )
legend( 'RMSE' , 'Dynamic range RMSE')
filename = sprintf('./imgs/rmse_test.png');
saveas(fig, filename, 'png');

% Classification;
cls_model = fitcknn( y_rep_train' , train_face_id );
prediction = predict( cls_model , y_rep_test' );

% Measure success rates
counter = 0;
correct = 0;
for i = 1 : 1 : 20
    if face_id( i ) ~= 0 && face_id( i ) ~= -1
        if prediction( i ) == face_id( i )
            correct = correct + 1;
        end
        counter = counter + 1;
    end
end
cls_success_precent = 100 * correct / counter;

return;