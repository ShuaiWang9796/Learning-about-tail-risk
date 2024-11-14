classdef ExpectileRegressionLayer < nnet.layer.RegressionLayer
    properties
        Tau
    end

    methods
        function layer = ExpectileRegressionLayer(name, tau)
            layer.Name = name;
            layer.Tau = tau;
            layer.Description = "Expectile Regression Layer - Tau: " + num2str(tau);
        end

        function loss = forwardLoss(layer, Y, T)
            errors = T - Y;
            loss = abs(layer.Tau - (errors <= 0)) .* (errors).^2;
            loss = mean(mean(loss));
        end

        function dLdY = backwardLoss(layer, Y, T)
            Y = permute(Y, [2 1 3]);
            T = permute(T, [2 1 3]);

            N = numel(Y);
            errors = T - Y;

            indicator = abs(layer.Tau - (errors <= 0));
            dLdY = -2 * errors .* indicator / N;

            dLdY = permute(dLdY, [2 1 3]);
        end
    end
end