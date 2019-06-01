


%% call for the given script
inl1;

%close all;

%% compute the camera matrix P using the Direct Linear Transform - (a)

% 1. Computing the matrix A

% points where 'NaN' exists - don't take into account

[A, nan_arr] = build_A_matrix (pts2d,pts3d);


% 2. use the SVD

[U,D,V] = svd(A);

% 3. P is the minimal right eigenvector.

P = V(:,12);
P = [P(1:4,1)';P(5:8,1)';P(9:12,1)'];

%% 4. verification - answers the q. (b)

% Estimate the error on the points expect of 9,10,11 (NaN)
% we use the MSE to estimate the error

Err_mean = calculate_MSE(pts2d, pts3d, P, nan_arr);

fprintf('-------------------------------\n');
fprintf('TOTAL Error value (MSE): %2.4f\n\n',Err_mean);


%% 5. Determine parameters K, R, t - (c)

% P = [ KR | Kt ] 

K_t = P(:,4);
K_R = P(:,1:3);


% Using the QR factorization provided in hw we find the K and R matrixes
[K,R] = rq(K_R);

% find the translation vector t using the K matrix just found
t = K^(-1)*K_t;
R_t = [R t];  %  P = K [ R | t]

% print the Projection Matrix as a multiplication of the intrinsic
% parameters, and extrinsic parameters (rotation, translation)

fprintf('Intrinsic parameters (K matrix):\n');
disp(K);

fprintf('Intrinsic parameters:\n');


fprintf('Extrinsic parameters (Rt matrix):\n');
fprintf('R:\n');
disp(R);
fprintf('t:\n');
disp(t);




%% (f) 6. Calculate the translation vector t in the world coordinate system

c = -R^(-1)*t; 


%% (g) Plotting the camera location

% plot as well the border of the frame of the camera.
% Using the coordinates on the image in 2d

% frame_2d = [0 size(im,1) size(im,1) 0;
%             0 0 size(im,2) size(im,2);
%             1 1 1 1];
%         
% frame_3d = get_3D_points(frame_2d, P);


% getting the camera center (0,0,0,1) in the camera coordinate system
% We use the translation in the world coordinate system that was calculated

figure(2);
grid;
plot3(c(1),c(2),c(3),'r*');


function Points_3D = get_3D_points (points_2D, P)

    % Using the reprojection matrix, calculate the 3D points
    Points_3D = [];
    
    for num_point=1:size(points_2D,2)
        P_pseudo = (P'*P)^(-1)*P';
        Points_3D(:,end+1) = P_pseudo*points_2D(:,num_point);
        Points_3D(:,end) = Points_3D(:,end)./Points_3D(4,end);
    end

end


function Err_mean = calculate_MSE (pts2d, pts3d, P, nan_arr)

    num_points = size(pts2d,2);
    num_used_points = 0;
    Err_total = 0;

    for point_num=1:num_points

        fprintf('\nEstimating error on a point: %2.0f\n', point_num)

        if (~isempty(find(nan_arr==point_num,1)))
            fprintf('Point has NaN values in 2D image. Ignoring\n');
            continue;
        end



        coor_g = P*pts3d(:,point_num);
        x_g = [coor_g(1)/coor_g(3),coor_g(2)/coor_g(3)]; % g for genearated
        x_v = [pts2d(1,point_num), pts2d(2,point_num)]; % v for verification

        fprintf('Reprojected coords: X: %2.4f Y: %2.4f \n', x_g(1), x_g(2))
        fprintf('Real coords:       X: %2.4f Y: %2.4f \n', x_v(1), x_v(2))

        Err_local = sqrt(sum((x_g-x_v).^2));
        fprintf('Error value (SE):    %2.4f\n\n',Err_local);
        Err_total = Err_total+Err_local;

        num_used_points = num_used_points+1;
    end

    Err_mean = Err_total/num_used_points;



end



function [A, nan_arr] = build_A_matrix (pts2d,pts3d)

    num_points = size(pts2d,2);
    non_nan_pts = 1;
    nan_arr = [];

    for point_num=1:num_points

        if ( isnan(pts2d(1,point_num)) ||  isnan(pts2d(1,point_num)) )
           fprintf('point_num %2.0f has NaN values, ignoring \n', point_num);
           nan_arr(end+1) = point_num;
           continue;
        end

        row_num = 1+(non_nan_pts-1)*2;

        A(row_num,:) = [pts3d(:,point_num)'...
                        zeros(1,4)...
                        -pts3d(:,point_num)'*pts2d(1,point_num)];


        A(row_num+1,:) = [zeros(1,4)...
                          pts3d(:,point_num)'...
                          -pts3d(:,point_num)'*pts2d(2,point_num)];

        non_nan_pts = non_nan_pts +1;
    end
end



