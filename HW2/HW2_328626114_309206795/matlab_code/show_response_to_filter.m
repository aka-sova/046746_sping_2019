function show_response_to_filter(image,filter)

% input is 224* 224 * 3
% filter is 3 by 3 by 3
% output is 224* 224 * 1

image_filtered = imfilter(image, double(filter));

% we take only the middle layer
image_filtered = image_filtered(:,:,2);

% displaying the result
fig = figure();
imshow(image_filtered);

end

