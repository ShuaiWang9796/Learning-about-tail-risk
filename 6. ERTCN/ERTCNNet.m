function lgraph = ERTCNNet(numFilters,filterSize,dropoutFactor,numBlocks,numFeatures,outputFeatures,alpha)
    %%  Input layer structure
    layer = sequenceInputLayer(numFeatures, Normalization = "rescale-symmetric", Name = "input");
    
    %%  Add the input layer to the blank network
    lgraph = layerGraph(layer);
    outputName = layer.Name;
    %%  Establish the network structure -the residual block
    for i = 1 : numBlocks
        dilationFactor = 2 ^(i - 1);                                                                % 
        layers = [
    
            convolution1dLayer(filterSize, numFilters, DilationFactor = dilationFactor, ...         
            Padding = "causal", Name="conv1_" + i)                                                  % 
            layerNormalizationLayer                                                                 % 
    
            spatialDropoutLayer(dropoutFactor)                                                      %
    
            convolution1dLayer(filterSize, numFilters, ...                                          
            DilationFactor = dilationFactor, Padding = "causal")                                    % 
            layerNormalizationLayer                                                                 % 
            reluLayer                                                                               % 
    
            spatialDropoutLayer(dropoutFactor)                                                      % 
            
            additionLayer(2, Name = "add_" + i)];                                                   % 
    
        lgraph = addLayers(lgraph, layers);                          % 
        lgraph = connectLayers(lgraph, outputName, "conv1_" + i);    % 
    
        % res
        if i == 1
    
            layer = convolution1dLayer(1, numFilters, Name = "convSkip");    % 
            lgraph = addLayers(lgraph, layer);                               % 
            lgraph = connectLayers(lgraph, outputName, "convSkip");          % 
            lgraph = connectLayers(lgraph, "convSkip", "add_" + i + "/in2"); % 
    
        else
            
            lgraph = connectLayers(lgraph, outputName, "add_" + i + "/in2"); % 
        
        end
        
        % update the name
        outputName = "add_" + i;
    
    end
    
    %%  output layer
    layers = [
        fullyConnectedLayer(outputFeatures, Name = "fc")
        ExpectileRegressionLayer('expRegLayer', alpha)]; % expectile regression layer
    
    lgraph = addLayers(lgraph, layers);                % 
    lgraph = connectLayers(lgraph, outputName, "fc");  % 
    
    %%  print the network structure
    figure
    plot(lgraph)
    title("Temporal Convolutional Network")
    set(gcf,'color','w')
    
end