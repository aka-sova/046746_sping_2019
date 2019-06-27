

%% calculate the fisher basis


% run the question 1 to load its data
run('Main.m');
close all;

% reorganize the images into cells, to fit the FLD function

% face_data - cell array containing the face images

for i=1:size(image,3)
    face_data{i} = image(:,:,i);    
end

% label_train - the training labels matching c classes
label_train = train_face_id;

C = 15;

[meanvec, basis] = fisherface(face_data, label_train, c);

