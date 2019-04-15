function [Reconstructed] = Style_Trans(Pic_1,Pic_2)
% Style transfer function. Gets 2 pictures, picture 2 style applies to
% picture 1 and returns as Reconstructed

Pic_1 = double(Pic_1);
Pic_2 = double(Pic_2);

% Size of pyramid
n = 6;

epsilon = 1e-4;

% Creating the LoG pyramid, from it computing the energy for each image and
% the gain parameter
for i = 1 : 1 : 3
    
    [~, LoG_I(:,:,i,:)] = LoG_creator(Pic_1(:,:,i), 6);    
    [~, LoG_E(:,:,i,:)] = LoG_creator(Pic_2(:,:,i), 6);
    
    for j = 1 : 1 : n
        sigma = 2 ^ (j+1);
        g_filter = fspecial('gaussian', 4 * sigma, sigma);
        Energy_I(:,:,i,j) = imfilter(LoG_I(:,:,i,j) .^ 2, g_filter);
        Energy_E(:,:,i,j) = imfilter(LoG_E(:,:,i,j) .^ 2, g_filter);
        gain(:,:,i,j) = ( Energy_E(:,:,i,j) ./ ( Energy_I(:,:,i,j) + epsilon ) ) .^.5;
    end
    
end

% for i = 1 : 1 : n
%     figure(i + 100)
%     imshow(uint8(Energy_E(:,:,:,i)))
% end
% 
% for i = 1 : 1 : n
%     figure(i + 150)
%     imshow(uint8(Energy_I(:,:,:,i)))
% end
% 
% for i = 1 : 1 : n
%     figure(i + 200)
%     imshow(uint8(gain(:,:,:,i)))
% end

% Apply the conditions for gain
idx = gain > 2.8;
gain(idx) = 2.8;

idx = gain < 0.9;
gain(idx) = 0.9;

% Creating the gain modified LoG image for each matrix
for i = 1 : 1 : 3
    
    for j = 1 : 1 : n
        
        if j ~= n
            LcO(:,:,i,j) = gain(:,:,i,j) .* LoG_I(:,:,i,j);
        else
            LcO(:,:,i,j) = LoG_E(:,:,i,j);
        end
        
    end
    
    LcO_reduced(:,:,:) = LcO(:,:,i,:);
    Reconstructed(:,:,i) = Image_Recreator(LcO_reduced);
    
end
    
% for k = 1 : 1 : n
%     figure(k + 400)
%     imshow(uint8(LcO(:,:,:,k)))
% end

end

