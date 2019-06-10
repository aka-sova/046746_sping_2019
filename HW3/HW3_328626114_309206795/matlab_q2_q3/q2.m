


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

[Err_mean, Err_local, pts2d_reprojected] = calculate_MSE(pts2d, pts3d, P, nan_arr);

reproject_sum = [pts2d(1:2,:)',pts2d_reprojected',Err_local'];

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
R_t = [R t];  %  P = K [ R | t ]

% print the Projection Matrix as a multiplication of the intrinsic
% parameters, and extrinsic parameters (rotation, translation)

fprintf('Intrinsic parameters (K matrix):\n');
disp(K);

fprintf('Intrinsic parameters (normalized):\n');
K_norm = K/K(3,3);
disp(K_norm);

fprintf('Extrinsic parameters (Rt matrix):\n');
fprintf('R:\n');
disp(R);
fprintf('t:\n');
disp(t);

%% (e) verify the orientation of the matrix

Z_dir_world = [0; 0; 1];
Cam_dir = R*Z_dir_world;


%% (f) 6. Calculate the translation vector t in the world coordinate system

c = -R^(-1)*t; 


%% (g) Plotting the camera location


% getting the camera center (0,0,0,1) in the camera coordinate system
% We use the translation in the world coordinate system that was calculated

figure(2);
grid;
plot3(c(1),c(2),c(3),'r*');
txt = {'Camera location'};
text(c(1),c(2),c(3),txt)

% plot the 4 edges
dist_frames = 30;

x_end = size(im,2);
y_end = size(im,1);

frame_loc_2d(:,1) = [0; 0];
frame_loc_2d(:,2) = [x_end; 0];
frame_loc_2d(:,3) = [x_end; y_end];
frame_loc_2d(:,4) = [0; y_end];
w = 0.0005;

frame_loc_2d_normed = frame_loc_2d;
frame_loc_2d_normed(3,:) = 1;
frame_loc_2d_normed = frame_loc_2d_normed*w;

for point_num=1:size(frame_loc_2d_normed,2)

    frame_3d(:,point_num) = (K^-1)*frame_loc_2d_normed(:,point_num);
    frame_3d(:,point_num) = R^(-1)*frame_3d(:,point_num);
    frame_3d_normed(:,point_num) = frame_3d(:,point_num)/norm(frame_3d(:,point_num));
end

frame_3d_normed = frame_3d_normed*dist_frames;


%draw frames
% for ii=1:4
%     index=[ii,mod(ii,4)+1];
%     plot3(frame_3d_normed(1,index), ...
%         frame_3d_normed(2,index),...
%         frame_3d_normed(3,index),'b-');
% end

for ii=1:4
    index=[ii,mod(ii,4)+1];
    plot3([c(1) c(1)-frame_3d_normed(1,index)], ...
        [c(2) c(2)-frame_3d_normed(2,index)],...
        [c(3) c(3)-frame_3d_normed(3,index)],'b-');
end




%% (h) Plotting the ball vector

ball_loc_2d = [1602; 666];
w = 0.0005;

Distance = 80;

ball_loc_2d_normed = [ball_loc_2d;1]*w;

% using the pseudoinverse, we get big errors

% so we use the step by step way.

Vec_ball_W2 = (K^-1)*ball_loc_2d_normed;
Vec_ball_W2 = R^(-1)*Vec_ball_W2;
Vec_ball_W2_normed = Vec_ball_W2(1:3)/norm(Vec_ball_W2);

figure(2);
plot3([c(1) c(1)-Vec_ball_W2_normed(1)*Distance], ...
    [c(2) c(2)-Vec_ball_W2_normed(2)*Distance],...
    [c(3) c(3)-Vec_ball_W2_normed(3)*Distance],'-');



function Points_3D = get_3D_points (points_2D, P)

    % Using the reprojection matrix, calculate the 3D points
    Points_3D = [];
    
    for num_point=1:size(points_2D,2)
        P_pseudo = (P'*P)^(-1)*P';
        Points_3D(:,end+1) = P_pseudo*points_2D(:,num_point);
        Points_3D(:,end) = Points_3D(:,end)./Points_3D(4,end);
    end

end


function [Err_mean, Err_local, x_g] = calculate_MSE (pts2d, pts3d, P, nan_arr)

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
        x_g(:,point_num) = [coor_g(1)/coor_g(3),coor_g(2)/coor_g(3)]; % g for genearated
        x_v = [pts2d(1,point_num), pts2d(2,point_num)]; % v for verification

        fprintf('Reprojected coords: X: %2.4f Y: %2.4f \n', x_g(1,point_num), x_g(2,point_num))
        fprintf('Real coords:       X: %2.4f Y: %2.4f \n', x_v(1), x_v(2))

        Err_local(point_num) = sqrt(sum((x_g(:,point_num)'-x_v).^2));
        fprintf('Error value (SE):    %2.4f\n\n',Err_local(point_num));
        Err_total = Err_total+Err_local(point_num);

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



