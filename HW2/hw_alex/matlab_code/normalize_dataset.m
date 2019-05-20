function [normed] = normalize_dataset(dataset)

    % function normalizes the dataset
    % rows are samples
    % columns are parameters
    % 
    % we normalize each parameter column
    
    normed = [];
    
    for col_num=1:size(dataset,2)
        
        min_col_val = min(dataset(:,col_num));
        max_col_val = max(dataset(:,col_num));
        
        normed(:,col_num) = (dataset(:,col_num)-min_col_val) ./ ...
            (max_col_val - min_col_val);
    end
    
end

