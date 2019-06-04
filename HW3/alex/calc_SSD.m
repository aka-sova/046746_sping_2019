function [min_x, min_y, SSD_out] = calc_SSD (image,template)

    % this function calcs SSD of image [N,M,3]
    % and a template [w,w,3]
    
    % output size is [N, M, 1]
    
    % calculate SSD for each channel separately
    % then find the mean SSD between all of them
    
    M = size(image,1);
    N = size(image,2);
    
    m = size(template,1);
    n = size(template,2);
    z = m*n; % length of template
    
    temp_dist = (m-1)/2; % how wide is template
    
    
    
    fun = @(block_struct) sum((block_struct.data-local_temp).^2);
    
    for channel=1:size(image,3)
        
        local_im = image(:,:,channel);
        local_im = padarray(local_im,[0 temp_dist],0,'both');
        local_temp = template(:,:,channel);
        

        
        
        image_block_ordered = double(im2col(local_im,'indexed', size(local_temp), 'sliding'));
        temp_ordered = double(reshape(local_temp,[z,1]));
        
%         fun = @(block_struct) sum((block_struct.data-temp_ordered).^2);
%         SSD = blockproc(image_block_ordered,size(temp_ordered),fun);
        
        % other way around - duplicate the template
        temp_duplicated = repmat(temp_ordered, 1, N);
        SSD = sum((image_block_ordered-temp_duplicated).^2);
        
        
        SSD_total(1,:,channel) = reshape(SSD, [1, N]);
        
    end

    SSD_out = mean(SSD_total,3);
    
    [~, min_idx] = min(SSD_out(:));
    
    % column is x
    % row is y
    
    [min_y, min_x] = ind2sub(size(SSD_out),min_idx);
    
    
end

