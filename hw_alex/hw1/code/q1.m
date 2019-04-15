

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









