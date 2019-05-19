function vizualize_filter ( filter )
    % fitler is 3X3X3
    
    % 1. normalize the weights to fit in [0 1]
    f_min = min(filter(:));
    f_max = max(filter(:));
    filter_normed = (filter - f_min) / (f_max-f_min );
    
    filter_uint = im2uint8(filter_normed);
    
    fig = figure('Position',[200 200 600 300]);
    for filt_depth_lvl=1:size(filter_uint,2)
        subplot(1,3,filt_depth_lvl);
        imshow(filter_uint(:,:,filt_depth_lvl));
    end
    % title('Vizualization of a filter');
    
    
    
end

