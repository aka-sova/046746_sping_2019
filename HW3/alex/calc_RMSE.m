function rmse = calc_RMSE (im_1,im_2)
    rmse = sqrt(mean((im_1(:) - im_2(:)).^2));  % Root Mean Squared Error
end

