

%% calculate the fisher basis


% run the question 1 to load its data

mkdir('imgs2');

run('readYaleFaces.m');
test_image = zeros( 243 , 320 , 20 );

for i = 1:20 % The image numbering is bad programing; put all images in a single variable
    testFileName = ['image' num2str(i)];
    test_image( : , : , i ) = double( eval ( testFileName ) );
end

clearvars image*

image = reshape( A , [ 243 , 320 , 150 ]);


N = size(image,3);

% reorganize the images into cells, to fit the FLD function

% face_data - cell array containing the face images

for i=1:size(image,3)
    face_data{i} = image(:,:,i);    
end

% label_train - the training labels matching c classes
label_train = train_face_id;

c = 15;
% Calculate the fisher basis
[meanvec, basis] = fisherface(face_data, label_train, c);

% project all the images on the Fisher space, calculate reconstruction
% error

average_image = zeros( 243 , 320 );
for i = 1 : 1 : 150  
    average_image = average_image + double( image( : , : , i ) ) / 150;
end

subtracted_image = zeros( 243 , 320 );

for i = 1 : 1 : 150
    subtracted_image( : , : , i ) = ( image( : , : , i ) - average_image );
end

basis_matrix = reshape( basis , 243 * 320 , c-1 );

for i = 1 : 1 : 150
    
    y_rep_train__fish( : , i ) = basis_matrix' * reshape( subtracted_image( : , : , i ) , 243 * 320 , 1 );
    reconstructed = reshape( average_image , [ 320 * 243 , 1] ) + basis_matrix * y_rep_train__fish( : , i );
    reconstructed = reshape( reconstructed , [ 243 , 320 ] );
    
    normalized_training_image = image( : , : , i ) / ( max( image( : , : , i ) ) - min( image( : , : , i ) ) );
    normalized_reconstruction = reconstructed / ( max( reconstructed ) - min( reconstructed ) );
    
    Dr_RMSE_squared( i ) = sum(sum( ( normalized_reconstruction...
        - normalized_training_image ).^2 )) / ( 243 * 320 );
    
    RMSE_squared( i ) = sum(sum( ( reconstructed / 255 ...
        - image( : , : , i ) / 255 ).^2 )) / ( 243 * 320 );
    
end


Dr_RMSE = Dr_RMSE_squared.^.5;
RMSE = RMSE_squared.^.5;

fig = figure(310);
hold on
plot(RMSE, 'r')
plot(Dr_RMSE)
grid;
xlabel( 'Image number' )
ylabel( 'Error' )
legend( 'RMSE' , 'Dynamic range RMSE')
filename = sprintf('./imgs2/rmse_train.png');
saveas(fig, filename, 'png');




clearvars Dr_RMSE_squared RMSE_squared Dr_RMSE RMSE

subtracted_test_image = zeros( 243 , 320 , 20 );
for i = 1 : 1 : 20
    subtracted_test_image( : , : , i ) = test_image( : , : , i ) - average_image;
end

y_rep_test_fish = zeros( c-1 , 20 );

for i = 1 : 1 : 20
    
    y_rep_test_fish( : , i ) = basis_matrix' * reshape( subtracted_test_image( : , : , i ) , 243 * 320 , 1 );
    reconstructed = reshape( average_image , [ 320 * 243 , 1] ) + basis_matrix * y_rep_test_fish( : , i );
    reconstructed = reshape( reconstructed , [ 243 , 320 ] );

    normalized_test_image = test_image( : , : , i ) / ( max( test_image( : , : , i ) ) - min( test_image( : , : , i ) ) );
    normalized_reconstruction = reconstructed / ( max( reconstructed ) - min( reconstructed ) );
    
    Dr_RMSE_squared( i ) = sum(sum( ( normalized_reconstruction...
        - normalized_test_image ).^2 )) / ( 243 * 320  );
    
    RMSE_squared( i ) = sum(sum( ( reconstructed / 255 ...
        - test_image( : , : , i ) / 255 ).^2 )) / ( 243 * 320 );
    
end

Dr_RMSE = Dr_RMSE_squared.^.5;
RMSE = RMSE_squared.^.5;

fig = figure(410);
hold on
plot(RMSE, 'r')
plot(Dr_RMSE)
grid;
xlabel( 'Image number' )
ylabel( 'Error' )
legend( 'RMSE' , 'Dynamic range RMSE')
filename = sprintf('./imgs2/rmse_test.png');
saveas(fig, filename, 'png');






% Classification;
cls_model = fitcknn( y_rep_train__fish' , train_face_id );
prediction = predict( cls_model , y_rep_test_fish' );

% Measure success rates
counter = 0;
correct = 0;
for i = 1 : 1 : 20
    if face_id( i ) ~= 0 && face_id( i ) ~= -1
        if prediction( i ) == face_id( i )
            correct = correct + 1;
            answer = 'correct';
        else
            answer = 'false;';
        end
        
        fprintf('Face num. %1.0f the classification is %s\n', i, answer);
        counter = counter + 1;
    end
end
cls_success_precent = 100 * correct / counter;

fprintf('Classification rate is %2.2f percent \n', cls_success_precent);












