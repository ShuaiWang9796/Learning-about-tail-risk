function [VaR, ES] = Qpredict_risk(beta0, gamma, alpha, beta0_Train, gamma_Train, Y1, k)
sigma = (1/k) * ones(size(gamma));
beta = [zeros(size(gamma, 1), 1), gamma]; 
beta = beta(:,1:end-1);
beta(:,1) = beta0(:,1);
beta = (gamma-beta)./(2*sigma);
temp1 = [zeros(size(beta, 1), 1), beta];
beta = beta - temp1(:,1:end-1);
beta(:,end) = gamma(:,end) - sum(beta(:,1:end-1), 2);

ksi = cumsum(sigma, 2);
ksi = [zeros(size(ksi, 1), 1), ksi]; 
ksi = ksi(:,1:end-1);

alpha1 = optimize_alpha(alpha, beta0_Train, gamma_Train, Y1, k);
alpha2 = alpha1 - (1-alpha1*10);
indices = ksi < alpha2;
pred = sum(beta0*alpha2, 2);
VaR = pred + sum(((alpha2-ksi).^2).*beta.*indices, 2);

alpha0001 = optimize_alpha(0.001, beta0_Train, gamma_Train, Y1, k);
alpha_low = alpha0001 - (1-alpha0001*10);
range1 = alpha_low:0.001:alpha2;
V_temp = [];
for i = 1:length(range1)
    indices = ksi < range1(i);
    pred = sum(beta0*range1(i), 2);
    V_temp(:, i) = pred + sum(((range1(i)-ksi).^2).*beta.*indices, 2);
end
ES = mean(V_temp, 2);
end