clear
close all
clc

%% First image set

image_I = imread('data/Inputs/imgs/0004_6.png');
image_E = imread('data/Examples/imgs/6.png');
background = imread('data/Examples/bgs/6.jpg');
mask = imread('data/Inputs/masks/0004_6.png');

Reconstructed = Style_Trans(image_I, image_E);
Reconstructed = Background_Applier(Reconstructed,  background, mask);

figure(1)
imshow(Reconstructed)
title('Style transfer: example pics')

%% Second image set

my_pic = imread('data/Inputs/imgs/my_pic');
my_pic = imresize(my_pic, [900, 1600]);
vg_sn = imread('data/Inputs/imgs/starry_night.jpg');
vg_sn = imresize(vg_sn, [900, 1600]);

Reconstructed = Style_Trans(my_pic, vg_sn);

figure(2)
imshow(Reconstructed)
title('Style transfer: custom pictures')

%% Third image set

image_I = imread('data/Inputs/imgs/0004_6.png');
image_E = imread('data/Examples/imgs/16.png');
image_E2 = imread('data/Examples/imgs/21.png');

Reconstructed = Style_Trans(image_I, image_E);
Reconstructed = Background_Applier(Reconstructed,  background, mask);
figure(3)
imshow(Reconstructed)


Reconstructed = Style_Trans(image_I, image_E2);
Reconstructed = Background_Applier(Reconstructed,  background, mask);
figure(4)
imshow(Reconstructed)

%% Fourth image set

image_I = imread('data/Inputs/imgs/0006_001.png');
image_E = imread('data/Examples/imgs/0.png');
image_E2 = imread('data/Examples/imgs/9.png');
image_E3 = imread('data/Examples/imgs/10.png');
mask = imread('data/Inputs/masks/0006_001.png');

Reconstructed = Style_Trans(image_I, image_E);
Reconstructed = Background_Applier(Reconstructed,  background, mask);
figure(5)
imshow(Reconstructed)

Reconstructed = Style_Trans(image_I, image_E2);
Reconstructed = Background_Applier(Reconstructed,  background, mask);
figure(6)
imshow(Reconstructed)

Reconstructed = Style_Trans(image_I, image_E3);
Reconstructed = Background_Applier(Reconstructed,  background, mask);
figure(7)
imshow(Reconstructed)


return;