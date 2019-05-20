clear 
close all
clc

%% Add folders to path

addpath( genpath( 'CNN Images' ) );
addpath( genpath( 'sift' ) );
addpath( genpath( 'Stop Images' ) );

%% Read images

Stop_1 = imread('StopSign1.jpg');
%Stop_1 = double(Stop_1);
Stop_2 = imread('StopSign2.jpg');
%Stop_2 = double(Stop_2);
Stop_3 = imread('StopSign3.jpg');
%Stop_3 = double(Stop_3);
Stop_4 = imread('StopSign4.jpg');
%Stop_4 = double(Stop_4);

%% Q1 sift features

[ frames_1 , desc_1 ] = sift( rgb2gray( Stop_1 ) );
figure( 101 )
marked = insertMarker( Stop_1 , [ frames_1( 1 , : ) ; frames_1( 2 , : ) ]' );
imshow( marked )

[ frames_2 , desc_2 ] = sift( rgb2gray( Stop_2 ) );
figure( 102 )
marked = insertMarker( Stop_2 , [ frames_2( 1 , : ) ; frames_2( 2 , : ) ]' );
imshow( marked )

[ frames_3 , desc_3 ] = sift( rgb2gray( Stop_3 ) );
figure( 103 )
marked = insertMarker( Stop_3 , [ frames_3( 1 , : ) ; frames_3( 2 , : ) ]' );
imshow( marked )

[ frames_4 , desc_4 ] = sift( rgb2gray( Stop_4 ) );
figure( 104 )
marked = insertMarker( Stop_4 , [ frames_4( 1 , : ) ; frames_4( 2 , : ) ]' );
imshow( marked )


%% Q2 sift matching

keypoints_2 = siftmatch( desc_1 , desc_2 );
keypoints_3 = siftmatch( desc_1 , desc_3 );
keypoints_4 = siftmatch( desc_1 , desc_4 );

figure(202)
plotmatches( double(Stop_1)/255 , double(Stop_2)/255 , frames_1 , frames_2 , keypoints_2 )

figure(203)
plotmatches( double(Stop_1)/255 , double(Stop_3)/255 , frames_1 , frames_3 , keypoints_3 )

figure(204)
plotmatches( double(Stop_1)/255 , double(Stop_4)/255 , frames_1 , frames_4 , keypoints_4 )

%% Q4

[ hom_2_max , in_2 ] = RANSAC_matching( frames_1 , frames_2 , keypoints_2 );
frames_match_21 = frames_1( : , keypoints_2( 1 , : ) );
frames_match_22 = frames_2( : , keypoints_2( 2 , : ) );
figure(402)
plotmatches( double(Stop_1)/255 , double(Stop_2)/255 , frames_match_21 , frames_match_22 , [ in_2 ; in_2 ] )

[ hom_3_max , in_3 ] = RANSAC_matching( frames_1 , frames_3 , keypoints_3 );
frames_match_31 = frames_1( : , keypoints_3( 1 , : ) );
frames_match_32 = frames_3( : , keypoints_3( 2 , : ) );
figure(403)
plotmatches( double(Stop_1)/255 , double(Stop_3)/255 , frames_match_31 , frames_match_32 , [ in_3 ; in_3 ] )

[ hom_4_max , in_4 ] = RANSAC_matching( frames_1 , frames_4 , keypoints_4 );
frames_match_41 = frames_1( : , keypoints_4( 1 , : ) );
frames_match_42 = frames_4( : , keypoints_4( 2 , : ) );
figure(404)
plotmatches( double(Stop_1)/255 , double(Stop_4)/255 , frames_match_41 , frames_match_42 , [ in_4 ; in_4 ] )

%% Q5

int_2 = affine_transformation( Stop_1 , hom_2_max );
figure(502)
imshow( uint8( int_2 ) )

int_3 = affine_transformation( Stop_1 , hom_3_max );
figure(503)
imshow( uint8( int_3 ) )

int_4 = affine_transformation( Stop_1 , hom_4_max );
figure(504)
imshow( uint8( int_4 ) )

%% Q6

stitch_2 = stitch( Stop_2 , int_2 );
figure(602)
imshow( uint8( stitch_2 ) )

stitch_3 = stitch( Stop_3 , int_3 );
figure(603)
imshow( uint8( stitch_3 ) )

stitch_4 = stitch( Stop_4 , int_4 );
figure(604)
imshow( uint8( stitch_4 ) )

return;