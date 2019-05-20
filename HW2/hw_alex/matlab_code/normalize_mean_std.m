function output = normalize_mean_std(im, req_mean, req_std)

    % input the mean and var of the size of num of channels in image
    ch_num = size(im);

    for i=1:ch_num(3)
        input_im = im(:,:,i);
        
        mean_input = mean(input_im(:));
        mean_std = std(input_im(:));
        
        im(:,:,i) = ((im(:,:,i) - mean_input).*(req_std(i)/mean_std))+req_mean(i);
    end
    
    output = im;
end

