


addpath './Depth with stereo data/groundtruth'
addpath './Depth with stereo data/input'


% test the SSD

% image = imread('test03_l.png');
% 
% template_coords = [54 55];
% 
% template = image(template_coords(1)-1:template_coords(1)+1,...
%                 template_coords(2)-1:template_coords(2)+1, :);
%             
% [min_x, min_y, SSD] = calc_SSD(image, template);
% 
% fprintf('Chosen coords: [%2.0f %2.0f]\n', template_coords(1), template_coords(2));
% fprintf('Found match: [%2.0f %2.0f]\n', min_x, min_y);






% test the correspondence of the points
% find matching points between the images
% 
% I_left = imread('test03_l.png');
% I_right = imread('test03_r.png');
% template_size = 11;
% X_left_coords = [190 161; 308 90; 128 195];
% X_right_coords = find_matching_pixels(I_left,I_right,X_left_coords, template_size);
% 
% 
% 
% 
% 
% figure(1)
% imshow(I_left);
% hold all;
% 
% figure(2)
% imshow(I_right);
% hold all;
% 
% for pixel_num = 1:size(X_left_coords,1)
%     
%     figure(1);
%     plot(X_left_coords(pixel_num,1),X_left_coords(pixel_num,2),'*');
%     
%     figure(2);
%     plot(X_right_coords(pixel_num,1),X_right_coords(pixel_num,2),'*');
%     
% end


%%

template_sizes = [5 11 21];
im_names = {'03' '04' '07'};



formatOut = 'dd_hh_mm';
t = datestr(datetime('now','TimeZone','Asia/Jerusalem'),formatOut);
mkdir(strcat('results/',t));

DisparityMax = 60;

for im_idx = 1:length(im_names)
    
    
    im_name = im_names{im_idx};
    

    left_im_addr = strcat('input/test',im_name,'_l.png');
    right_im_addr = strcat('input/test',im_name,'_r.png');
    
    I_left = imread(left_im_addr);
    I_right = imread(right_im_addr);
    
%     I_left = imresize(I_left,0.3);
%     I_right = imresize(I_right,0.3);
    
    for temp_size_idx=1 : length(template_sizes)

        template_size = template_sizes(temp_size_idx);
        fprintf('Initialize disparity: im %2.0f window: %2.0f\n', im_idx, template_size);
        local_disp_map = uint8(find_disparity(I_left, I_right, template_size, DisparityMax));
        disparity_map{im_idx}(:,:,temp_size_idx) = local_disp_map;
        fprintf('Disparity found\n');


        % disparity_map = uint8(disparity_map);

        fig1 = figure();
        title_text = sprintf(strcat('Disparity map for image ',...
            im_name,' window size = ', num2str(template_size)));
        imshow(disparity_map{im_idx}(:,:,temp_size_idx),[]);
        title(title_text);
        
        % save the data
        out_filename = strcat('results/',t,'/disp_map_',im_name,'_template_',num2str(temp_size_idx),'.mat');
        save(out_filename, 'local_disp_map');
        
        fig_filename = strcat('results/',t,'/disp_map_',im_name,'_template_',num2str(temp_size_idx),'.png');
        saveas(fig1,fig_filename)
        close(fig1);
    end
end












