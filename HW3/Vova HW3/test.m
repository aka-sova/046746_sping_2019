clear
close all
clc

gif = imread('seq.gif', 'Frames', 'all');

%% Initialization

gif = double(gif);
gif_size = size(gif);

K = 16;
t_limit = 1;

x_parts = ceil( gif_size(1) / K );
y_parts = ceil( gif_size(2) / K );

%% Computing optical flow

optical_flow_matrix = zeros( x_parts , y_parts , gif_size( 4 ) , 2 );

for t = 1 : 1 : gif_size( 4 ) - 1
    for idx_x = 0 : 1 : x_parts - 1
        for idx_y = 0 : 1 : y_parts - 1
        
        
            A_matrix = zeros( K^2 , 2 );
            b_vector = zeros( K^2 , 1 );

            for k_x = idx_x * K + 1 : 1 : K * ( 1 + idx_x )
                for k_y = idx_y * K + 1 : 1 : K * ( 1 + idx_y )


                    % Computing Ix
                    if k_x == 1 + idx_x * K
                        A_matrix( k_x + ( k_y - 1 ) * K , 1 ) =...
                            ( gif( 2 , k_y , 1 , t ) - gif( 1 , k_y , 1 , t ) ) / 1;
                    elseif k_x == K + idx_x * K
                        A_matrix( k_x + ( k_y - 1 ) * K , 1 ) =...
                            ( gif( K , k_y , 1 , t ) - gif( K - 1 , k_y , 1 , t ) ) / 1;
                    else
                        A_matrix( k_x + ( k_y - 1 ) * K , 1 ) =...
                            gif( k_x + 1 , k_y , 1 , t ) / 2 - gif( k_x - 1 , k_y , 1 , t ) / 2;
                    end

                    % Computing Iy
                    if k_y == 1 + idx_y * K
                        A_matrix( k_x + ( k_y - 1 ) * K , 2 ) =...
                            ( gif( k_x , 2 , 1 , t ) - gif( k_x , 1 , 1 , t ) ) / 1;
                    elseif k_y == K + idx_y * K
                        A_matrix( k_x + ( k_y - 1 ) * K , 2 ) =...
                            ( gif( k_x , K , 1 , t ) - gif( k_x , K - 1 , 1 , t ) ) / 1;
                    else
                        A_matrix( k_x + ( k_y - 1 ) * K , 2 ) =...
                            gif( k_x , k_y + 1 , 1 , t ) / 2 - gif( k_x , k_y - 1 , 1 , t ) / 2;
                    end

                    % Computing It
                    b_vector( k_x + ( k_y - 1 ) * K ) =...
                        ( gif( k_x , k_y , 1 , t + 1 ) - gif( k_x , k_y , 1 , t ) ) / 1;

                end
            end

            % Computing A^T*A eigenvalues
            eigen_AtA = eig( A_matrix' * A_matrix ); 

            % Computing optical flow
            if min( eigen_AtA ) >= t_limit * 256
                optical_flow_matrix( idx_x + 1 , idx_y + 1 , t , : ) = ( A_matrix' * A_matrix )^(-1) * A_matrix' * b_vector;
            end
        
        end
    end
end

return;