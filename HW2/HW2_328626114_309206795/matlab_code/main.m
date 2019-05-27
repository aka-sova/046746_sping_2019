



% part 1: read the bird images
% we need to transform the images so they would fit into the VGG16 input
% which is 224*224
% then normalize the inputs:
%   a. first to double ([0 1])
%   b. then to mean and var values specified for VGG16 input network
%       mean = [0.485, 0.456, 0.406]
%       std = [0.229, 0.224, 0.225]
% normalization is already done for us
% 

% we simply rescale them 

net = vgg16();

bird_pics = dir('CNN images/birds');
bird_num = 0;
   
%% question 2: display the birds and get predictions
for i=1:length(bird_pics)

   if  (~strcmp(bird_pics(i).name, '.') && ~strcmp(bird_pics(i).name, '..'))
       fig = figure();
       bird = imread(strcat('CNN images/birds/',bird_pics(i).name));
       
       title(strcat('Input bird ', int2str(bird_num)));
       bird_num = bird_num +1;
       
       bird_resized = imresize(bird,[224 224]);
       imshow(bird_resized);
       
%        bird_norm_1 = im2double(bird_resized);
%        
%        req_mean = [0.485 0.456 0.406];
%        req_std = [0.229, 0.224, 0.225];
%        
%        bird_norm_2 = normalize_mean_std(bird_norm_1, req_mean, req_std);
       
       label = classify(net, bird_resized);
       text(10, 20, char(label),'Color','white')
       
       
   end    
end



%% question 3: input new image from internet and classify

bear = imread('bear.jpg');
fig = figure();
imshow(bear);

bear_resized = imresize(bear,[224 224]);
fig = figure();
imshow(bear_resized);
title('Bear image');
label = classify(net, bear_resized);
certainty = get_prediction_certainty(net, bear_resized); 
text(10, 20, char(label),'Color','white')
text(10, 40, strcat('Certainty: ', num2str(certainty, '% 10.2f'), ' [%]'),'Color','white')


%% question 4a: Using a geometric transformation - shear
a = 0.45;
T = maketform('affine', [1 0 0; a 1 0; 0 0 1] );
bear_sheared = imtransform(bear,T,'cubic','FillValues',[0 0 0]');
fig = figure();
imshow(bear_sheared);
title('Bear after geometric transformation');
% resize and get prediction

bear_sheared_resized = imresize(bear_sheared,[224 224]);
fig = figure();
imshow(bear_sheared_resized);
title('Bear after geometric transformation');
label = classify(net, bear_sheared_resized);
certainty = get_prediction_certainty(net, bear_sheared_resized); 
text(10, 40, strcat('Certainty: ', num2str(certainty, '% 10.2f'), ' [%]'),'Color','white')
text(10, 20, char(label),'Color','white')


%% question 4b: using a color transformation - affect the hue layer
bear_hsv = rgb2hsv(bear);
bear_hsv(:,:,1) = bear_hsv(:,:,1)*2.5; % affect Hue
bear_colored = im2uint8(hsv2rgb(bear_hsv));
fig = figure();
imshow(bear_colored);

bear_hue_resized = imresize(bear_colored,[224 224]);
fig = figure();
imshow(bear_hue_resized);
title('Bear after color transformation');
label = classify(net, bear_hue_resized);
certainty = get_prediction_certainty(net, bear_hue_resized); 
text(10, 40, strcat('Certainty: ', num2str(certainty, '% 10.2f'), ' [%]'),'Color','white')
text(10, 20, char(label),'Color','white')


%% question 4c: using a filter on an image. Lets create a filter of a motion blur
H = fspecial('motion',20,45);
bear_blurred = imfilter(bear,H,'replicate');
imshow(bear_blurred);

bear_blurred_resized = imresize(bear_blurred,[224 224]);
fig = figure();
imshow(bear_blurred_resized);
title('Bear after motion blur filter');
label = classify(net, bear_blurred_resized);
certainty = get_prediction_certainty(net, bear_blurred_resized); 
text(10, 40, strcat('Certainty: ', num2str(certainty, '% 10.2f'), ' [%]'),'Color','white')
text(10, 20, char(label),'Color','white')



%% question 5: Show the 2 filters response to the 3 input images of the bear

% load the original bear image
bear = imread('bear.jpg');
bear_resized = imresize(bear,[224 224]);

% choose some random filters
filter_1 = net.Layers(2).Weights(:,:,:,5);
filter_2 = net.Layers(2).Weights(:,:,:,32);

% vizualizing them
vizualize_filter(filter_1)
vizualize_filter(filter_2)

% showing the response
show_response_to_filter(bear_resized, filter_1)
show_response_to_filter(bear_resized, filter_2)


%% answering the question - showing the response for 3 transformations

show_response_to_filter(bear_sheared_resized, filter_1)
show_response_to_filter(bear_sheared_resized, filter_2)

show_response_to_filter(bear_hue_resized, filter_1)
show_response_to_filter(bear_hue_resized, filter_2)

show_response_to_filter(bear_blurred_resized, filter_1)
show_response_to_filter(bear_blurred_resized, filter_2)



%% question 6 - vizualizing all the dogs and cats feature vectors

net = vgg16();

input_imgs{1}.files_list = dir('CNN images/cats');
input_imgs{1}.fold_name = 'CNN images/cats/';
input_imgs{1}.name = 'cats';


input_imgs{2}.files_list = dir('CNN images/dogs');
input_imgs{2}.fold_name = 'CNN images/dogs/';
input_imgs{2}.name = 'dogs';


features = [];
features_cats=[];
features_dogs=[];
for j=1:length(input_imgs)
    
    
    for i=1:length(input_imgs{j}.files_list)
        
        if  (~strcmp(input_imgs{j}.files_list(i).name, '.') && ...
                ~strcmp(input_imgs{j}.files_list(i).name, '..'))

            input_image = imread(strcat(input_imgs{j}.fold_name, ...
                input_imgs{j}.files_list(i).name));

            % resize the image to pass to the network
            im_resized = imresize(input_image,[224 224]);
            
            fprintf('Analyzing %s: %2.0f\n', input_imgs{j}.name, i+j-2);
            
            % get the fc7 layer response
            features(end+1,:) = activations(net,im_resized,'fc7');
            
            if (strcmp(input_imgs{j}.name, 'cats'))
                features_cats(end+1,:) = features(end,:);
            end
            
            if (strcmp(input_imgs{j}.name, 'dogs'))
                features_dogs(end+1,:) = features(end,:);
            end
            
            
        end    
    end
end


% we have to normalize the data to bring everything on the same scale

% feat_normed = normalize_dataset(features);
% feat_standartized = standartize_dataset(features);

%   features_mean = features - mean(features); % bring all to 0

[coeff, score, latent] = pca(features);

cats_PC_scores = features_cats*coeff;
dogs_PC_scores = features_dogs*coeff;


fig = figure();
scatter3(cats_PC_scores(:,1),cats_PC_scores(:,2),cats_PC_scores(:,3),'r')
hold on;
scatter3(dogs_PC_scores(:,1),dogs_PC_scores(:,2),dogs_PC_scores(:,3),'b')
axis equal
grid on
legend('Cats','Dogs');
xlabel('1st Principal Component')
ylabel('2nd Principal Component')
zlabel('3rd Principal Component')

fig = figure();
plot(cats_PC_scores(:,1),cats_PC_scores(:,2),'ro')
hold on;
plot(dogs_PC_scores(:,1),dogs_PC_scores(:,2),'bo')
axis equal
legend('Cats','Dogs');
grid on;
xlabel('1st Principal Component')
ylabel('2nd Principal Component')


%% question 7 - vizualizing cat and dog iamge from the internet on pca graph


cat_internet = imread('cat.jpg');
cat_resized = imresize(cat_internet,[224 224]);
cat_feat  = activations(net,cat_resized,'fc7');
cat_PC_score  = cat_feat*coeff;

dog_internet = imread('dog.jpg');
dog_resized = imresize(dog_internet,[224 224]);
dog_feat  = activations(net,dog_resized,'fc7');
dog_PC_score  = dog_feat*coeff;

% plot the dots on the existing graph
fig = figure();
scatter3(cats_PC_scores(:,1),cats_PC_scores(:,2),cats_PC_scores(:,3),'r')
hold on;
scatter3(dogs_PC_scores(:,1),dogs_PC_scores(:,2),dogs_PC_scores(:,3),'b')
hold on;
scatter3(cat_PC_score(:,1),cat_PC_score(:,2),cat_PC_score(:,3),'k', 'filled')
hold on;
scatter3(dog_PC_score(:,1),dog_PC_score(:,2),dog_PC_score(:,3),'g', 'filled')
axis equal
grid on
legend('Cats','Dogs','New cat','New dog');
xlabel('1st Principal Component')
ylabel('2nd Principal Component')
zlabel('3rd Principal Component')

fig = figure();
plot(cats_PC_scores(:,1),cats_PC_scores(:,2),'ro')
hold on;
plot(dogs_PC_scores(:,1),dogs_PC_scores(:,2),'bo')
axis equal
plot(cat_PC_score(:,1),cat_PC_score(:,2),'.k','MarkerSize',30)
axis equal
plot(dog_PC_score(:,1),dog_PC_score(:,2),'.g','MarkerSize',30)
axis equal
legend('Cats','Dogs','New cat','New dog');
grid on;
xlabel('1st Principal Component')
ylabel('2nd Principal Component')



% finding the distant from the nearest neighbor. 
% we use the euclidean distance
nearest_cat = find_closest_neighbor(cats_PC_scores, cat_PC_score); % 9
% nearest cat is No. 8 (imgs starts count from 0)

nearest_dog = find_closest_neighbor(dogs_PC_scores, dog_PC_score); % 6
% nearest cat is No. 5 (imgs starts count from 0)


%% question 8 - use the wolf and tiger images

tiger_internet = imread('tiger.jpg');
tiger_resized = imresize(tiger_internet,[224 224]);
tiger_feat  = activations(net,tiger_resized,'fc7');
tiger_PC_score  = tiger_feat*coeff;

wolf_internet = imread('wolf.jpg');
wolf_resized = imresize(wolf_internet,[224 224]);
wolf_feat  = activations(net,wolf_resized,'fc7');
wolf_PC_score  = wolf_feat*coeff;

% plot the dots on the existing graph
fig = figure();
scatter3(cats_PC_scores(:,1),cats_PC_scores(:,2),cats_PC_scores(:,3),'r')
hold on;
scatter3(dogs_PC_scores(:,1),dogs_PC_scores(:,2),dogs_PC_scores(:,3),'b')
hold on;
scatter3(tiger_PC_score(:,1),tiger_PC_score(:,2),tiger_PC_score(:,3),'k', 'filled')
hold on;
scatter3(wolf_PC_score(:,1),wolf_PC_score(:,2),wolf_PC_score(:,3),'g', 'filled')
axis equal
grid on
legend('Cats','Dogs','Tiger','Wolf');
xlabel('1st Principal Component')
ylabel('2nd Principal Component')
zlabel('3rd Principal Component')

fig = figure();
plot(cats_PC_scores(:,1),cats_PC_scores(:,2),'ro')
hold on;
plot(dogs_PC_scores(:,1),dogs_PC_scores(:,2),'bo')
axis equal
plot(tiger_PC_score(:,1),tiger_PC_score(:,2),'.k','MarkerSize',30)
axis equal
plot(wolf_PC_score(:,1),wolf_PC_score(:,2),'.g','MarkerSize',30)
axis equal
legend('Cats','Dogs','Tiger','Wolf');
grid on;
xlabel('1st Principal Component')
ylabel('2nd Principal Component')




% finding the distant from the nearest neighbor. 
% we use the euclidean distance
nearest_cat = find_closest_neighbor(cats_PC_scores, tiger_PC_score); % 8
% nearest cat is No. 7 (imgs starts count from 0)

nearest_dog = find_closest_neighbor(dogs_PC_scores, wolf_PC_score); % 8
% nearest cat is No. 7 (imgs starts count from 0)


%% additional material (out of scope):

% layers = [39];
% 
% channels = 1:2;
% 
% for layer = layers
%     I = deepDreamImage(net,layer,channels, ...
%         'Verbose',false, ...
%         'NumIterations',50);
%     
%     figure
%     I = imtile(I,'ThumbnailSize',[128 128]);
%     imshow(I)
%     name = net.Layers(layer).Name;
%     title(['Layer ',name,' Features'])
% end



%%  PCA good lesson

% 
% rng 'default'
% M = 7; % Number of observations
% N = 5; % Number of variables observed
% X = rand(M,N);
% % De-mean
% X = X - mean(X);
% % Do the PCA
% [coeff,score,latent] = pca(X);
% % Calculate eigenvalues and eigenvectors of the covariance matrix
% covarianceMatrix = cov(X);
% [V,D] = eig(covarianceMatrix);
% % "coeff" are the principal component vectors. These are the eigenvectors of the covariance matrix. Compare ...
% coeff
% V
% % Multiply the original data by the principal component vectors to get the projections of the original data on the
% % principal component vector space. This is also the output "score". Compare ...
% dataInPrincipalComponentSpace = X*coeff
% score
% % The columns of X*coeff are orthogonal to each other. This is shown with ...
% corrcoef(dataInPrincipalComponentSpace)
% % The variances of these vectors are the eigenvalues of the covariance matrix, and are also the output "latent". Compare
% % these three outputs
% var(dataInPrincipalComponentSpace)'
% latent
% sort(diag(D),'descend')












