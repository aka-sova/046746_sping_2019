clear all
close all
clc

%image = imread('building.jpg');

%% Part 1 example code
f=zeros(101,101);
f(1,1)=1;
H=hough(f);

figure(1)
imshow(H,[])
f(101,1)=1;
H=hough(f);

figure(2)
imshow(H,[])
f(1,101)=1;
H=hough(f);

figure(3)
imshow(H,[])
f(101,101)=1;

figure(4)
H=hough(f);
imshow(H,[])
f(51,51)=1;

return