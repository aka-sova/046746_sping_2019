clear
close all
clc



%% part 1 - hough lines 

f=zeros(101,101);
f(1,1)=1;
H=hough(f);
imshow(H,[])
f(101,1)=1;
H=hough(f);
imshow(H,[])
f(1,101)=1;
H=hough(f);
imshow(H,[])
f(101,101)=1;
H=hough(f);
imshow(H,[])
f(51,51)=1;




[H,T,R] = hough(f);
imshow(H,'XData',T,'YData',R,...
      'InitialMagnification','fit');
title('Hough transform #4');
xlabel('\theta'), ylabel('\rho');
axis on, axis normal, hold on;


%% part 2 

close all;

% (b) 
% create a square
f=ones(201,201);

% padding it with zeroes
fb =  padarray(f,[100,100],'both');
f_binary = imbinarize(fb);  % binarize the image
fig_21 = figure(21);
imshow(f_binary,[])
title('binary image of a square');


% (c)

% using the Carry edge detector to generate image of edges
BW = edge(f_binary);
fig_22 = figure(22);
imshow(BW,[])
title('Edges of a binary square');


% (d) 

% finding the hough transform

[H,T,R] = hough(BW,'RhoResolution',0.5,'ThetaResolution',0.1);
fig23 = figure(23);
imshow(H,[],'InitialMagnification','fit');
title('Original Hough transform of a square'); 
xlabel('\theta'), ylabel('\rho');
axis on, axis normal, hold on;


fig24 = figure(24);
imshow(imadjust(rescale(H)),'XData',T,'YData',R,...
      'InitialMagnification','fit');
title('Rescaled & adjusted Hough transform of a square');
xlabel('\theta'), ylabel('\rho');
axis on, axis normal, hold on;


% (f)

% finding the peaks
peaks = houghpeaks(H,10); % setting maximum amount of peaks to 10

theta_values = T(peaks(:,2));
rho_values = R(peaks(:,1));

fprintf('the pairs of the peaks in the square are the following:\n\n');
for i=1:length(theta_values) 
    fprintf('\ttheta : %2.0f   rho: %2.0f\n', theta_values(i), rho_values(i));    
end

% (g)

% reproducing the lines on the image itself using a dedicated function
fig25 = figure(25);
imshow(fb);
hold on;
draw_hough_lines(fb,theta_values,rho_values,'green');
title('Original image with Hough lines (n=4)');






%% part 3


% (g) 

building = imread('building.jpg');
% 1. plot the image to see what we have
fig31 = figure(31);
imshow(building,[]);
% we see that it is grayscale already.
% 2. use the canny edge detector to detect edges. The following parameters
% showed quiet good results.

building_edges = edge(building,'Canny',[0.05 0.1]);
fig32 = figure(32);
imshow(building_edges,[]);

% 3. Finding the Hough transform of the image 
[H,T,R] = hough(building_edges,'RhoResolution',0.5,'ThetaResolution',0.1);
fig33 = figure(33);
imshow(imadjust(rescale(H)),'XData',T,'YData',R,...
      'InitialMagnification','fit');
title('Rescaled & adjusted Hough transform of a building');
xlabel('\theta'), ylabel('\rho');
axis on, axis normal, hold on;

% 4. Finding the peaks of the Hough map
peaks = houghpeaks(H,100,'threshold',30); 
theta_values = T(peaks(:,2));
rho_values = R(peaks(:,1));

fprintf('\nDetected %2.0f peaks (lines) in the image \n\n', length(theta_values));

fig34 = figure(34);
imshow(building);
hold on;
draw_hough_lines(building,theta_values,rho_values,'blue');
title('Original image with Hough lines (n=100)');











%% First image set, example images

image_I = imread('data/Inputs/imgs/0004_6.png');
image_E = imread('data/Examples/imgs/6.png');
background = imread('data/Examples/bgs/21.jpg');
mask = imread('data/Inputs/masks/0004_6.png');

Reconstructed = Style_Trans(image_I, image_E);
Reconstructed = Background_Applier(Reconstructed,  background, mask);

figure(1001)
imshow(Reconstructed)
title('Style transfer: example pics')

%% Second image set, custom images

my_pic = imread('data/Inputs/imgs/my_pic');
my_pic = imresize(my_pic, [900, 1600]);
vg_sn = imread('data/Inputs/imgs/starry_night.jpg');
vg_sn = imresize(vg_sn, [900, 1600]);

Reconstructed = Style_Trans(my_pic, vg_sn);

figure(1002)
imshow(Reconstructed)
title('Style transfer: custom pictures')

%% Third image set, style transfer, woman

image_I = imread('data/Inputs/imgs/0004_6.png');
image_E = imread('data/Examples/imgs/16.png');
image_E2 = imread('data/Examples/imgs/21.png');

Reconstructed = Style_Trans(image_I, image_E);
Reconstructed = Background_Applier(Reconstructed,  background, mask);
figure(1003)
imshow(Reconstructed)


Reconstructed = Style_Trans(image_I, image_E2);
Reconstructed = Background_Applier(Reconstructed,  background, mask);
figure(1004)
imshow(Reconstructed)

%% Fourth image set, style transfer, man

image_I = imread('data/Inputs/imgs/0006_001.png');
image_E = imread('data/Examples/imgs/0.png');
image_E2 = imread('data/Examples/imgs/9.png');
image_E3 = imread('data/Examples/imgs/10.png');
mask = imread('data/Inputs/masks/0006_001.png');

Reconstructed = Style_Trans(image_I, image_E);
Reconstructed = Background_Applier(Reconstructed,  background, mask);
figure(1005)
imshow(Reconstructed)

Reconstructed = Style_Trans(image_I, image_E2);
Reconstructed = Background_Applier(Reconstructed,  background, mask);
figure(1006)
imshow(Reconstructed)

Reconstructed = Style_Trans(image_I, image_E3);
Reconstructed = Background_Applier(Reconstructed,  background, mask);
figure(1007)
imshow(Reconstructed)


return;