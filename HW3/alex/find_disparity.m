function disparity_map = find_disparity(I_left,I_right, template_size, DisparityMax)

    sum_of_pixels = size(I_left,1)*size(I_left,2);
    counter = 0;
    counter_tot = 0;

    temp_dist = (template_size-1)/2; % how wide is template
    
    % image coordinate system:
    %
    % O----> x
    % |
    % |
    % v 
    % y
    
    for y=1:size(I_left,1)
        % y spans the rows
        for x=1:size(I_left,2)
            % x spans the cols
            X_left_coords = [x y];
            
            dist_to_border = calc_dist(I_left, y, x);
            
            if (dist_to_border < temp_dist+1 )
                % if template does not fit - fill the disparity with 0
                disparity_map(y,x) = 0;
            else
                X_right_coords = find_matching_pixels(I_left,I_right,X_left_coords, template_size);
                
                disparity = norm(X_right_coords-X_left_coords);
                
                if disparity < 0 
                    disparity_map(y,x) = 0;
                elseif disparity > DisparityMax
                    disparity_map(y,x) = DisparityMax;
                else
                    disparity_map(y,x) = disparity;
                end
                
                
            end
            
            counter = counter + 1;
            counter_tot = counter_tot+1;
            
            if (counter/sum_of_pixels > 0.1)
                fprintf('Done: %2.0f / %2.0f\n', counter_tot, sum_of_pixels);
                counter = 0;
            end
            

        end
    end

end


function distance = calc_dist(image, row, col)

    im_rows = size(image,1);
    im_cols = size(image,2);

    distance = min([row, im_rows-row, col, im_cols-col]);

end
