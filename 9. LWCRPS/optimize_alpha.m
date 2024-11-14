function best_param = optimize_alpha(tau, beta0_Train, gamma_Train, Y1, k)
func_to_minimize = @(z) double(quantileloss(tau, Qpredict_risk_VaR(beta0_Train, gamma_Train, z, k) , Y1));
% Define the parameter search range
lower_bound = 0;
upper_bound = 0.1;
% step size
step_size = 0.001;
% 
min_value = Inf;
best_param = lower_bound;
% 
for param = lower_bound:step_size:upper_bound
    % 
    value = func_to_minimize(param);    
    % 
    if value < min_value
        min_value = value;
        best_param = param;
    end
end
% results
fprintf('min value: %.4f\n', min_value);
fprintf('corresponding parameter: %.4f\n', best_param);
end