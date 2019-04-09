function [reconstructed_image] = reconstruct_from_log(log_pyramid)

image_x_size = length(log_pyramid{1}(:,1));
image_y_size = length(log_pyramid{1}(1,:));

reconstructed_image = uint8(zeros(image_x_size, image_y_size));

for layer_number=1:length(log_pyramid)
    reconstructed_image = reconstructed_image + log_pyramid{layer_number};
end

