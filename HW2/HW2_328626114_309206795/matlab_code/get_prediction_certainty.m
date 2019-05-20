function [accuracy] = get_prediction_certainty (net, input)

    % return the certainty of the prediction
    
    features = activations(net,input,'prob');
    [max_val, idx] = max(features);    
    accuracy = max_val * 100;
    
    
end

