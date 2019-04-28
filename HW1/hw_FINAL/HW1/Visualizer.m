clear
close all
clc

background = imread('data/Examples/bgs/21.jpg');
mask = imread('data/Inputs/masks/0004_6.png');
image = imread('data/Inputs/imgs/0004_6.png');

improved = Background_Applier(image, background, mask);
imshow(improved)

image2 = imread('building.jpg');
LoG = LoG_creator(image2, 4);

for i = 1 : 1 : 4
    figure(i)
    imshow(uint8(LoG(:,:,i)))
end

return;