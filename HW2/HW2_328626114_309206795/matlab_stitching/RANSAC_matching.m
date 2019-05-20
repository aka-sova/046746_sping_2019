function [max_hom_matrix , keypoint_inliers] = RANSAC_matching(frames_1, frames_2, keypoints)

% RANSAC for matching between keyframes in 2 images. The function gets 2
% matrices of keyframes, one per each image, and a set of matching indices.
% The function outputs the affine matrix with the maximal number of
% inliers.

    frames_match_1 = frames_1( : , keypoints( 1 , : ) );
    frames_match_2 = frames_2( : , keypoints( 2 , : ) );

    num_of_iterations = 1000;

    max_num_inliers = 0;

    for i = 1 : 1 : num_of_iterations

        indices = randperm( length(frames_match_1) , 6 ); % Random roll of matchin indices
        hom_matrix = affine_matches( frames_match_1 , frames_match_2 , [ indices ; indices ] );
        proj_err_all_points = [ frames_match_2( 1:2 , : ) ; ones( 1 , length(frames_match_1) ) ] -... % L2 norm for each pair
            hom_matrix * [ frames_match_1( 1:2 , : ) ; ones( 1 , length(frames_match_1) ) ];
        norm_err_all_points = vecnorm( proj_err_all_points );
        num_inlier = sum(norm_err_all_points < 5); % Number of inliers
        if max_num_inliers <= num_inlier && 3 <= num_inlier
            max_num_inliers = num_inlier;
            max_hom_matrix = affine_matches( frames_match_1 , frames_match_2 , ...
                [ find( norm_err_all_points < 5 ) ; find( norm_err_all_points < 5 ) ] ); % Saving the matrix with the maximal number of inliers
            keypoint_inliers = find( norm_err_all_points < 5 ); % Saving the indices of inliers
        end

    end

    
    
end

