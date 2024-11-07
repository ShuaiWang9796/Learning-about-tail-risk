function [VaR,ES] = hHistoricalVaRES(Sample,VaRLevel)
    % Compute historical VaR and ES
    % See [7] for technical details

    % Convert to losses
    Sample = -Sample;
    
    N = length(Sample);
    k = ceil(N*VaRLevel);
    
    z = sort(Sample);
    
    VaR = z(k);
    
    if k < N
       ES = ((k - N*VaRLevel)*z(k) + sum(z(k+1:N)))/(N*(1 - VaRLevel));
    else
       ES = z(k);
    end
end
