



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
   
% display the birds
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
       
       fprintf('The bird is 
       
       
   end    
end



% input new image from internet and classify

bear = imread('bear.jpg');
fig = figure();
imshow(bear);

bear_resized = imresize(bear,[224 224]);
fig = figure();
imshow(bear_resized);
title('Bear image');
label = classify(net, bear_resized);
text(10, 20, char(label),'Color','white')



% Using a geometric transformation - shear
a = 0.45;
T = maketform('affine', [1 0 0; a 1 0; 0 0 1] );
bear_sheared = imtransform(bear,T,'cubic','FillValues',[0 0 0]');
fig = figure();
imshow(bear_sheared);
title('Bear after geometric transformation');
% resize and get prediction

bear_resized = imresize(bear_sheared,[224 224]);
fig = figure();
imshow(bear_resized);
title('Bear after geometric transformation');
label = classify(net, bear_resized);
text(10, 20, char(label),'Color','white')


% using a color transformation - affect the hue layer
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
text(10, 20, char(label),'Color','white')


% using a filter on an image. Lets create a filter of a motion blur
H = fspecial('motion',20,45);
bear_blurred = imfilter(bear,H,'replicate');
imshow(bear_blurred);

bear_blurred_resized = imresize(bear_blurred,[224 224]);
fig = figure();
imshow(bear_blurred_resized);
title('Bear after motion blur filter');
label = classify(net, bear_blurred_resized);
text(10, 20, char(label),'Color','white')





% filter = net.Layers(2).Weights(:,:,:,:);
















