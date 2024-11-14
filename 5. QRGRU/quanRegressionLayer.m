


classdef quanRegressionLayer < nnet.layer.RegressionLayer
    % custom quantile regression layer
    properties
        tau
    end
    methods
        function layer = quanRegressionLayer(name,tau)
            layer.tau = tau;
            layer.Name = name;
            
            % Set layer description.
            layer.Description = 'quantile error';
        end
        
        function loss = forwardLoss(layer, Y, T)
            % Calculate quantile score.
  
            R = size(Y,1);
            quantileError = sum(max(layer.tau*(T-Y),(1-layer.tau)*(Y-T)))/R;  
            
            % Take mean over mini-batch.
            N = size(Y,3);
            loss = sum(quantileError)/N;
        end
        function dLdY = backwardLoss(layer,Y,T)
           
            dLdY =  single(-layer.tau*(T-Y>= 0) + (1-layer.tau) * (Y-T>=0));
                    
        end
        
        
    end
end


