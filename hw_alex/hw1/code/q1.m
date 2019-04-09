

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

% (b) 
% create a square
f=ones(201,201);

% padding it with zeroes
fb =  padarray(f,[100,100],'both');
f_binary = imbinarize(fb);  % binarize the image
fig_21 = figure();
imshow(f_binary,[])
title('binary image of a square');


% (c)

% using the Carry edge detector to generate image of edges
BW = edge(f_binary);
fig_22 = figure();
imshow(BW,[])
title('Edges of a binary square');


% (d) 

% finding the hough transform

[H,T,R] = hough(BW,'RhoResolution',0.5,'ThetaResolution',0.1);
fig23 = figure();
imshow(H,[],'InitialMagnification','fit');
title('Original Hough transform of a square'); 
xlabel('\theta'), ylabel('\rho');
axis on, axis normal, hold on;


fig24 = figure();
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















