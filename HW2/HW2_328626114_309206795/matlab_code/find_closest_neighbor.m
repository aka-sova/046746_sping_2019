function [closest_idx] = find_closest_neighbor(big_data, sample)

    % in both big_data, sample
    % rows are samples
    % columns are parameters
    
    
    
    dist_mx = big_data - sample;
    dist_mx = dist_mx.^2;
    
    dist_mx_sum = sum(dist_mx,2);
    dist_mx_sqrt = sqrt(dist_mx_sum);
    
    [~, closest_idx] = min(dist_mx_sqrt);
end

