function [meanvec, basis] = fisherface(face_data, label_train, c)
% FISHERFACE Creates the Fisher face basis for a set of images
%
% face_data - cell array containing the face images
% label_train - the training labels matching c classes
% c - number of classes to use
%
% meanvec - mean vector of the face images
% basis - Fisher basis of the face images
%


% 1. Use the algorithm you developed in the previous question 
%   to compute a basis with r= N-C components. Where N is the 
%   number of train images and C is a given parameter. This will be pca
%   W_pca


N = size(face_data,2);

R = N - c;

% reorder images to get 'subtracted_image':

average_image = zeros( 243 , 320 );
for i = 1 : 1 : 150
    average_image = average_image + double( face_data{i}(:,:) ) / 150; 
end

subtracted_image = zeros( 243 , 320 );
for i = 1 : 1 : 150    
    subtracted_image( : , : , i ) = ( face_data{i}(:,:) - average_image );
end

W_pca = eigenimages( subtracted_image / sqrt(150) , R );
W_matrix = reshape( W_pca , 243 * 320 , R );

% Project all the faces in the train set onto the pca W basis found on
%   (a) to find their low dimension
%   representations. Do not forget to subtract the mean face.

for i = 1 : 1 : 150 
    y_rep_train( : , i ) = W_matrix' * reshape(subtracted_image( : , : , i ) , 243 * 320 , 1 );
end


% Calculate the S_b and S_w

y_total_mean = mean(y_rep_train(:));

for class=1:c
   relevant_items = label_train==class;
   y_mean(:,class) = mean(y_rep_train(:,relevant_items),2); % [135 X 1]
   
   % inside class scatter
   
   y_mean_block = repmat(y_mean(:,class),1,10); % [135 X 10]
   
   S_w_class(class,:,:) = (y_rep_train(:,relevant_items)-y_mean_block)*...
       (y_rep_train(:,relevant_items)-y_mean_block)';
   

end


S_b = (y_mean-y_total_mean)*...
   (y_mean-y_total_mean)';  % [135 X 135]

S_w = reshape(sum(S_w_class,1), [135 135]); % [135 X 135]

% now find the W_fld using the eigs function. Find 'c-1' biggest
% eigenvalues and corresponding eigenvalues

[evecs, evals] = eigs(S_b, S_w, c-1);

% from the tutorial 11
W_fld = (evecs'*(S_w^(-1)*S_b))'; % [135 X 14]

W = W_matrix*W_fld;


basis = reshape(W, 243, 320 , c-1);
meanvec = mean(basis,3);

% plot all the basis:

% fig = figure();
% for im=1:c-1
%     subplot(5,3,im)
%     imshow(W_pca(:,:,im),[]);
% end































end

