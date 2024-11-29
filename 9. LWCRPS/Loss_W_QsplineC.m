function [loss,gradients] = Loss_W_QsplineC(net,X,T,k)
Y = forward(net,X);
labels = T;
beta0 = Y(1,:);
gamma = Y(2:k+1,:);

gamma = log(1+exp(gamma));
beta0 = log(1+exp(beta0));

gamma = dlarray(gamma, 'TB');
beta0 = dlarray(beta0, 'TB');
labels = dlarray(labels, 'TB');
% sigma
sigma = dlarray((1/k) * ones(size(gamma)));
% beta
beta = [zeros(size(gamma, 1), 1), gamma]; % 
beta = beta(:,1:end-1);
beta(:,1) = beta0(:,1);
beta = (gamma-beta)./(2*sigma);
temp1 = [zeros(size(beta, 1), 1), beta];
beta = beta - temp1(:,1:end-1);
beta(:,end) = gamma(:,end) - sum(beta(:,1:end-1), 2);
% calculate the maximum for each segment of the spline
ksi = dlarray([]);
for i = 1:length(labels)
    ksi(i,:) = arrayfun(@(n) sum(sigma(i, 1:n)), 1:width(sigma));
end
df1(1:k,:,:) = reshape(repmat(ksi, [size(sigma, 2), 1, 1]), [k, length(labels), k]);
df1 = dlarray(df1, 'TBS');
df2 = repmat(dlarray(ksi, 'TB'), [1, 1, 1]);
ksi =  [zeros(size(ksi, 1), 1), ksi];
ksi = ksi(:,1:end-1);

ksi1(1:k, :, :) = reshape(repmat(ksi, [k, 1, 1]), [k, length(labels), k]);
knots = df1 - ksi1;
knots(knots<0)=0;

beta00(1:k, :) = reshape(repmat(beta0, [k, 1]), [k, length(labels)]);
beta_temp(1:k,:,:) = reshape(repmat(beta, [k, 1, 1]), [k, length(labels), k]);

knots = df2 .* beta00 + reshape(sum((knots.^2) .* beta_temp, 3), [k, length(labels)]);
knots = dlarray(knots, 'TB');
knots = [zeros(size(knots, 1), 1), knots];
knots = knots(:,1:end-1);

diff = labels - knots;
alpha_l = diff > 0; 
alpha_A = sum(alpha_l.*beta, 2);
alpha_B = beta0(:, 1) - 2*sum(alpha_l.*beta.*ksi, 2);
alpha_C = -labels + sum(alpha_l.*beta.*ksi.*ksi, 2);

not_zero = (alpha_A~=0);
alpha = zeros(size(alpha_A));
idx = (alpha_B.^2 - 4*alpha_A.*alpha_C)<0;
diff = abs(diff);
index = (diff == min(diff, [], 2));
index(~idx, :) = 0;

alpha(idx) = ksi(index);
alpha(~not_zero) = -alpha_C(~not_zero)./alpha_B(~not_zero);
not_zero = ~(~not_zero | idx);
delta = (alpha_B(not_zero)).^2 - 4*alpha_A(not_zero).*alpha_C(not_zero);
alpha(not_zero) = (-alpha_B(not_zero) + sqrt(delta)) ./ (2*alpha_A(not_zero));

crps_1 = 2*labels.*(1/12 - (1-alpha.^3)/3 - alpha.^2 + alpha);
crps_2 = 2 * (1/4 - alpha + alpha.^2 - alpha.^3/3) + 2 * beta0 .* (1/20 - alpha.^2/2 + 2*alpha.^3/3 - alpha.^4/4);
crps_3 = sum(2 * alpha_l .* beta .* ((1-ksi).^2/3.*(alpha-ksi).^3 - (1-ksi)/2.*(alpha-ksi).^4 + ...
    (alpha-ksi).^5/5), 2);
crps_4 = sum(beta .* ((1-ksi).^6 ./30), 2);

crps = (crps_1 + crps_2 - crps_3 + crps_4);

loss = mean(crps);
gradients = dlgradient(dlarray(loss),net.Learnables);

end