function [recreated_image] = Image_Recreator(LoG_matrix)
% The function gets a double or uint8 LoG matrix, and outputs a uint8
% recreated image

    recreated_image = sum(LoG_matrix, 3);
    recreated_image = uint8(recreated_image);

end

