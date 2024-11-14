function VaR = Qpredict_risk_VaR(beta0, gamma, alpha, k)
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
alpha1 = alpha - (1-alpha*10);
indices = ksi < alpha1;
pred = sum(beta0*alpha1, 2);
VaR = pred + sum(((alpha1-ksi).^2).*beta.*indices, 2);

alpha_low = 0.001 - (1-0.001*10);
range1 = alpha_low:0.001:alpha1;
V_temp = [];
for i = 1:length(range1)
    indices = ksi < range1(i);
    pred = sum(beta0*range1(i), 2);
    V_temp(:, i) = pred + sum(((range1(i)-ksi).^2).*beta.*indices, 2);
end
ES = mean(V_temp, 2);
end