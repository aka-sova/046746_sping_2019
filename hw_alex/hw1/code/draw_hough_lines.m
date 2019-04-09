function [] = draw_hough_lines (image, theta_values, rho_values,color)


numLines = numel(rho_values);

% those are constant in all lines
x0 = 1;
xend = size(image,2); % image width


%// For each rho,theta pair...
for idx = 1 : numLines
    r = rho_values(idx); th = theta_values(idx); %// Get rho and theta

    %// if a vertical line, then draw a vertical line centered at x = r
    if (th == 0)
        yend = size(image,1); % image height
        line([r r], [1 yend], 'Color', color);
    else
        y0 = (-cosd(th)/sind(th))*x0 + (r / sind(th)); % theta [deg]
        yend = (-cosd(th)/sind(th))*xend + (r / sind(th));

        % add the line
        line([x0 xend], [y0 yend], 'Color',color);
   end
end

end

