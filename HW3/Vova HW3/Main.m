clear 
close all
clc

%% Part 1; display optical flow arrows in the middle of their designated square

[ gif , cmap ] = imread('seq.gif', 'Frames', 'all');

gif_size = size( gif );
K = 16;
t_limit = 1;

opt_flow_mat = optical_flow_creator( gif , K , t_limit );

figure(101)
display_time = 1;
imshow( gif( : , : , : , display_time ) );
hold on
x_mesh_cd = K / 2 : K : ceil( gif_size( 1 ) - K / 2 );
y_mesh_cd = K / 2 : K : ceil( gif_size( 2 ) - K / 2 );
[ x_grid , y_grid ] = meshgrid( x_mesh_cd , y_mesh_cd );
quiver( x_grid , y_grid , opt_flow_mat( : , : , display_time , 2 ) , opt_flow_mat( : , : , display_time , 1 ) , 'Color' , 'r' );
hold off

figure(102)
display_time = 10;
imshow( gif( : , : , : , display_time ) )
hold on
x_mesh_cd = K / 2 : K : ceil( gif_size( 1 ) - K / 2 );
y_mesh_cd = K / 2 : K : ceil( gif_size( 2 ) - K / 2 );
[ x_grid , y_grid ] = meshgrid( x_mesh_cd , y_mesh_cd );
quiver( x_grid , y_grid , opt_flow_mat( : , : , display_time , 2 ) , opt_flow_mat( : , : , display_time , 1 ) , 'Color' , 'r' );
hold off

figure(103)
display_time = 20;
imshow( gif( : , : , : , display_time ) )
hold on
x_mesh_cd = K / 2 : K : ceil( gif_size( 1 ) - K / 2 );
y_mesh_cd = K / 2 : K : ceil( gif_size( 2 ) - K / 2 );
[ x_grid , y_grid ] = meshgrid( x_mesh_cd , y_mesh_cd );
quiver( x_grid , y_grid , opt_flow_mat( : , : , display_time , 2 ) , opt_flow_mat( : , : , display_time , 1 ) , 'Color' , 'r' );
hold off

figure(104)
display_time = 30;
imshow( gif( : , : , : , display_time ) )
hold on
x_mesh_cd = K / 2 : K : ceil( gif_size( 1 ) - K / 2 );
y_mesh_cd = K / 2 : K : ceil( gif_size( 2 ) - K / 2 );
[ x_grid , y_grid ] = meshgrid( x_mesh_cd , y_mesh_cd );
quiver( x_grid , y_grid , opt_flow_mat( : , : , display_time , 2 ) , opt_flow_mat( : , : , display_time , 1 ) , 'Color' , 'r' );
hold off

figure(105)
display_time = 40;
imshow( gif( : , : , : , display_time ) )
hold on
x_mesh_cd = K / 2 : K : ceil( gif_size( 1 ) - K / 2 );
y_mesh_cd = K / 2 : K : ceil( gif_size( 2 ) - K / 2 );
[ x_grid , y_grid ] = meshgrid( x_mesh_cd , y_mesh_cd );
quiver( x_grid , y_grid , opt_flow_mat( : , : , display_time , 2 ) , opt_flow_mat( : , : , display_time , 1 ) , 'Color' , 'r' );
hold off

%% Part 3

K = 16;
t_limit = 0.1;

opt_flow_mat = optical_flow_creator( gif , K , t_limit );

figure(301)
display_time = 1;
imshow( gif( : , : , : , display_time ) )
hold on
x_mesh_cd = K / 2 : K : ceil( gif_size( 1 ) - K / 2 );
y_mesh_cd = K / 2 : K : ceil( gif_size( 2 ) - K / 2 );
[ x_grid , y_grid ] = meshgrid( x_mesh_cd , y_mesh_cd );
quiver( x_grid , y_grid , opt_flow_mat( : , : , display_time , 2 ) , opt_flow_mat( : , : , display_time , 1 ) , 'Color' , 'r' );

figure(302)
display_time = 10;
imshow( gif( : , : , : , display_time ) )
hold on
x_mesh_cd = K / 2 : K : ceil( gif_size( 1 ) - K / 2 );
y_mesh_cd = K / 2 : K : ceil( gif_size( 2 ) - K / 2 );
[ x_grid , y_grid ] = meshgrid( x_mesh_cd , y_mesh_cd );
quiver( x_grid , y_grid , opt_flow_mat( : , : , display_time , 2 ) , opt_flow_mat( : , : , display_time , 1 ) , 'Color' , 'r' );

figure(303)
display_time = 20;
imshow( gif( : , : , : , display_time ) )
hold on
x_mesh_cd = K / 2 : K : ceil( gif_size( 1 ) - K / 2 );
y_mesh_cd = K / 2 : K : ceil( gif_size( 2 ) - K / 2 );
[ x_grid , y_grid ] = meshgrid( x_mesh_cd , y_mesh_cd );
quiver( x_grid , y_grid , opt_flow_mat( : , : , display_time , 2 ) , opt_flow_mat( : , : , display_time , 1 ) , 'Color' , 'r' );

figure(304)
display_time = 30;
imshow( gif( : , : , : , display_time ) )
hold on
x_mesh_cd = K / 2 : K : ceil( gif_size( 1 ) - K / 2 );
y_mesh_cd = K / 2 : K : ceil( gif_size( 2 ) - K / 2 );
[ x_grid , y_grid ] = meshgrid( x_mesh_cd , y_mesh_cd );
quiver( x_grid , y_grid , opt_flow_mat( : , : , display_time , 2 ) , opt_flow_mat( : , : , display_time , 1 ) , 'Color' , 'r' );

figure(305)
display_time = 40;
imshow( gif( : , : , : , display_time ) )
hold on
x_mesh_cd = K / 2 : K : ceil( gif_size( 1 ) - K / 2 );
y_mesh_cd = K / 2 : K : ceil( gif_size( 2 ) - K / 2 );
[ x_grid , y_grid ] = meshgrid( x_mesh_cd , y_mesh_cd );
quiver( x_grid , y_grid , opt_flow_mat( : , : , display_time , 2 ) , opt_flow_mat( : , : , display_time , 1 ) , 'Color' , 'r' );

%% Part 4a

K = 8;
t_limit = 1;

opt_flow_mat = optical_flow_creator( gif , K , t_limit );

figure(401)
display_time = 1;
imshow( gif( : , : , : , display_time ) )
hold on
x_mesh_cd = K / 2 : K : ceil( gif_size( 1 ) - K / 2 );
y_mesh_cd = K / 2 : K : ceil( gif_size( 2 ) - K / 2 );
[ x_grid , y_grid ] = meshgrid( x_mesh_cd , y_mesh_cd );
quiver( x_grid , y_grid , opt_flow_mat( : , : , display_time , 2 ) , opt_flow_mat( : , : , display_time , 1 ) , 'Color' , 'r' );

figure(402)
display_time = 10;
imshow( gif( : , : , : , display_time ) )
hold on
x_mesh_cd = K / 2 : K : ceil( gif_size( 1 ) - K / 2 );
y_mesh_cd = K / 2 : K : ceil( gif_size( 2 ) - K / 2 );
[ x_grid , y_grid ] = meshgrid( x_mesh_cd , y_mesh_cd );
quiver( x_grid , y_grid , opt_flow_mat( : , : , display_time , 2 ) , opt_flow_mat( : , : , display_time , 1 ) , 'Color' , 'r' );

figure(403)
display_time = 20;
imshow( gif( : , : , : , display_time ) )
hold on
x_mesh_cd = K / 2 : K : ceil( gif_size( 1 ) - K / 2 );
y_mesh_cd = K / 2 : K : ceil( gif_size( 2 ) - K / 2 );
[ x_grid , y_grid ] = meshgrid( x_mesh_cd , y_mesh_cd );
quiver( x_grid , y_grid , opt_flow_mat( : , : , display_time , 2 ) , opt_flow_mat( : , : , display_time , 1 ) , 'Color' , 'r' );

figure(404)
display_time = 30;
imshow( gif( : , : , : , display_time ) )
hold on
x_mesh_cd = K / 2 : K : ceil( gif_size( 1 ) - K / 2 );
y_mesh_cd = K / 2 : K : ceil( gif_size( 2 ) - K / 2 );
[ x_grid , y_grid ] = meshgrid( x_mesh_cd , y_mesh_cd );
quiver( x_grid , y_grid , opt_flow_mat( : , : , display_time , 2 ) , opt_flow_mat( : , : , display_time , 1 ) , 'Color' , 'r' );

figure(405)
display_time = 40;
imshow( gif( : , : , : , display_time ) )
hold on
x_mesh_cd = K / 2 : K : ceil( gif_size( 1 ) - K / 2 );
y_mesh_cd = K / 2 : K : ceil( gif_size( 2 ) - K / 2 );
[ x_grid , y_grid ] = meshgrid( x_mesh_cd , y_mesh_cd );
quiver( x_grid , y_grid , opt_flow_mat( : , : , display_time , 2 ) , opt_flow_mat( : , : , display_time , 1 ) , 'Color' , 'r' );

%% Part 4b

K = 8;
t_limit = 0.1;

opt_flow_mat = optical_flow_creator( gif , K , t_limit );

figure(406)
display_time = 1;
imshow( gif( : , : , : , display_time ) )
hold on
x_mesh_cd = K / 2 : K : ceil( gif_size( 1 ) - K / 2 );
y_mesh_cd = K / 2 : K : ceil( gif_size( 2 ) - K / 2 );
[ x_grid , y_grid ] = meshgrid( x_mesh_cd , y_mesh_cd );
quiver( x_grid , y_grid , opt_flow_mat( : , : , display_time , 2 ) , opt_flow_mat( : , : , display_time , 1 ) , 'Color' , 'r' );

figure(407)
display_time = 10;
imshow( gif( : , : , : , display_time ) )
hold on
x_mesh_cd = K / 2 : K : ceil( gif_size( 1 ) - K / 2 );
y_mesh_cd = K / 2 : K : ceil( gif_size( 2 ) - K / 2 );
[ x_grid , y_grid ] = meshgrid( x_mesh_cd , y_mesh_cd );
quiver( x_grid , y_grid , opt_flow_mat( : , : , display_time , 2 ) , opt_flow_mat( : , : , display_time , 1 ) , 'Color' , 'r' );

figure(408)
display_time = 20;
imshow( gif( : , : , : , display_time ) )
hold on
x_mesh_cd = K / 2 : K : ceil( gif_size( 1 ) - K / 2 );
y_mesh_cd = K / 2 : K : ceil( gif_size( 2 ) - K / 2 );
[ x_grid , y_grid ] = meshgrid( x_mesh_cd , y_mesh_cd );
quiver( x_grid , y_grid , opt_flow_mat( : , : , display_time , 2 ) , opt_flow_mat( : , : , display_time , 1 ) , 'Color' , 'r' );

figure(409)
display_time = 30;
imshow( gif( : , : , : , display_time ) )
hold on
x_mesh_cd = K / 2 : K : ceil( gif_size( 1 ) - K / 2 );
y_mesh_cd = K / 2 : K : ceil( gif_size( 2 ) - K / 2 );
[ x_grid , y_grid ] = meshgrid( x_mesh_cd , y_mesh_cd );
quiver( x_grid , y_grid , opt_flow_mat( : , : , display_time , 2 ) , opt_flow_mat( : , : , display_time , 1 ) , 'Color' , 'r' );

figure(410)
display_time = 40;
imshow( gif( : , : , : , display_time ) )
hold on
x_mesh_cd = K / 2 : K : ceil( gif_size( 1 ) - K / 2 );
y_mesh_cd = K / 2 : K : ceil( gif_size( 2 ) - K / 2 );
[ x_grid , y_grid ] = meshgrid( x_mesh_cd , y_mesh_cd );
quiver( x_grid , y_grid , opt_flow_mat( : , : , display_time , 2 ) , opt_flow_mat( : , : , display_time , 1 ) , 'Color' , 'r' );
hold off


return;