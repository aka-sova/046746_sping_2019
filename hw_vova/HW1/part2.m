clear
close all
clc

%% First image set

% image_I = imread('data/Inputs/imgs/0004_6.png');
% image_E = imread('data/Examples/imgs/6.png');
% background = imread('data/Examples/bgs/6.jpg');
% mask = imread('data/Inputs/masks/0004_6.png');
% 
% image_I = double(image_I);
% image_E = double(image_E);

n = 6;

epsilon = 1e-4;

% for i = 1 : 1 : 3
%     
%     [~, LoG_I(:,:,i,:)] = LoG_creator(image_I(:,:,i), 6);    
%     [~, LoG_E(:,:,i,:)] = LoG_creator(image_E(:,:,i), 6);
%     
%     for j = 1 : 1 : n
%         sigma = 2 ^ (j+1);
%         g_filter = fspecial('gaussian', 4 * sigma, sigma);
%         Energy_I(:,:,i,j) = imfilter(LoG_I(:,:,i,j) .^ 2, g_filter);
%         Energy_E(:,:,i,j) = imfilter(LoG_E(:,:,i,j) .^ 2, g_filter);
%         gain(:,:,i,j) = ( Energy_E(:,:,i,j) ./ ( Energy_I(:,:,i,j) + epsilon ) ) .^.5;
%     end
%     
% end
% 
% % Apply the conditions for gain
% idx = gain > 2.8;
% gain(idx) = 2.8;
% 
% idx = gain < 0.9;
% gain(idx) = 0.9;
% 
% for i = 1 : 1 : 3
%     
%     for j = 1 : 1 : n
%         
%         if j ~= n
%             LcO(:,:,i,j) = gain(:,:,i,j) .* LoG_I(:,:,i,j);
%         else
%             LcO(:,:,i,j) = LoG_E(:,:,i,j);
%         end
%         
%     end
%     
%     LcO_reduced(:,:,:) = LcO(:,:,i,:);
%     Reconstructed(:,:,i) = Image_Recreator(LcO_reduced);
%     
% end
% 
% Reconstructed = Background_Applier(Reconstructed,  background, mask);
% 
% figure(1)
% imshow(Reconstructed)

%% Second image set

my_pic = imread('data/Inputs/imgs/my_pic');
my_pic = imresize(my_pic, [900, 1600]);
vg_sn = imread('data/Inputs/imgs/starry_night.jpg');
vg_sn = imresize(vg_sn, [900, 1600]);

my_pic = double(my_pic);
vg_sn = double(vg_sn);

for i = 1 : 1 : 3
    
    [~, LoG_MP(:,:,i,:)] = LoG_creator(my_pic(:,:,i), 6);    
    [~, LoG_VG(:,:,i,:)] = LoG_creator(vg_sn(:,:,i), 6);
    
    for j = 1 : 1 : n
        sigma = 2 ^ (j+1);
        g_filter = fspecial('gaussian', 4 * sigma, sigma);
        Energy_MP(:,:,i,j) = imfilter(LoG_MP(:,:,i,j) .^ 2, g_filter);
        Energy_VG(:,:,i,j) = imfilter(LoG_VG(:,:,i,j) .^ 2, g_filter);
        gain2(:,:,i,j) = ( Energy_VG(:,:,i,j) ./ ( Energy_MP(:,:,i,j) + epsilon ) ) .^.5;
    end
    
end

% Apply the conditions for gain
idx = gain2 > 2.8;
gain2(idx) = 2.8;

idx = gain2 < 0.9;
gain2(idx) = 0.9;

for i = 1 : 1 : 3
    
    for j = 1 : 1 : n
        
        if j ~= n
            LcO2(:,:,i,j) = gain2(:,:,i,j) .* LoG_MP(:,:,i,j);
        else
            LcO2(:,:,i,j) = LoG_VG(:,:,i,j);
        end
        
    end
    
    LcO_reduced(:,:,:) = LcO2(:,:,i,:);
    Reconstructed2(:,:,i) = Image_Recreator(LcO_reduced);
    
end

figure(2)
imshow(Reconstructed2)

return;