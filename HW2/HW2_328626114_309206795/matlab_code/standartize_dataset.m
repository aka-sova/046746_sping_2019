function [standartized] = standartize_dataset(dataset)

    % function normalizes the dataset
    % rows are samples
    % columns are parameters
    % 
    % we normalize each parameter column
    
    standartized = [];
    
    for col_num=1:size(dataset,2)
        
        mean_val = mean(dataset(:,col_num));
        std_val = std(dataset(:,col_num));
        
        standartized(:,col_num) = (dataset(:,col_num)-mean_val) ./ std_val;
    end
    
end

