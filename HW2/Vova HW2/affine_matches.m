function [hom_matrix] = affine_matches(frames_1, frames_2, keypoints)
    
% A function to get the affine matrix from 2 sets of frames from images
% with their matches. It gets keyframes from each image and their matching
% points according to SIFT (or any other algorithm) and outputs the 3x3
% homogenious matrix, affine version (no perspective)

frames_match_1 = frames_1( : , keypoints( 1 , : ) );
frames_match_2 = frames_2( : , keypoints( 2 , : ) );

for i = 1 : 1 : length( frames_match_1(1,:) )
    
    A( 2 * i - 1 : 2 * i , 1 : 6 ) = [ frames_match_1( 1 , i ) frames_match_1( 2 , i ) 1 0 0 0; ...
        0 0 0 frames_match_1( 1 , i ) frames_match_1( 2 , i ) 1];
    
    b( 2 * i - 1 : 2 * i ) = [ frames_match_2( 1 , i ) frames_match_2( 2 , i ) ];
    
end

hom_vector = ( A' * A ) ^ (-1) * A' * b';
hom_matrix = [ hom_vector( 1 : 3 )' ; hom_vector( 4 : 6 )' ; 0 0 1 ];

end

