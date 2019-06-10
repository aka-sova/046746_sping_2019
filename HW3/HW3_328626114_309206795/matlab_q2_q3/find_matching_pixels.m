function found_pixels = find_matching_pixels (img_1,img_2,given_pixels_1, template_size)

    % the function uses the SSD method to find corresponding pixels from a
    % second image given pixels in first image
    
    % given_pixels_1[:, x_coord y_coord] - refer to img_1!
    % found_pixels in same format
    
    % img_1, img_2 are uint8 [M*N*3]
    
    
    
    for pixel_num = 1: size(given_pixels_1,1)
        
        % 1. find the corresponding template
        
        % X coordinate is the column (2)
        % Y coordinate is the row (1)
        
        
        temp_dist = (template_size-1)/2; % how wide is template
        
        
        template = img_1(given_pixels_1(pixel_num,2)-temp_dist:given_pixels_1(pixel_num,2)+temp_dist,...
                        given_pixels_1(pixel_num,1)-temp_dist:given_pixels_1(pixel_num,1)+temp_dist, :);

        % pass only the relevant part of the image - using the rectified
        % constraint - the matching points have the same y coordinates
        
        y_coord = given_pixels_1(pixel_num,2);
        
        relevant_img_part = img_2(y_coord-temp_dist:y_coord+temp_dist,:,:);
                    
                    
        % get [row column] = [y x]
        [min_x, min_y, ~] = calc_SSD(relevant_img_part, template);
        
        % return pixels in [x y]
        found_pixels(pixel_num,:) = [min_x, y_coord];
        
    end
    
    
    
end

