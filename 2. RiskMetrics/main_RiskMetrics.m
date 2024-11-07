clc
clear
%% data loading
data = xlsread('SPXR.csv');
Returns = data;
SampleSize = length(Returns);
%% Define the parameters
pVaR = [0.05 0.025 0.01]; % three probability levels
EstimationWindowSize = 252;
testlen = length(data)-floor(0.75*length(data));
TestWindowStart = SampleSize - testlen + 1;
TestWindowEnd = SampleSize;
TestWindow = TestWindowStart:TestWindowEnd;
ReturnsTest = Returns(TestWindow);
Zscore = norminv(pVaR);
EWMA95 = zeros(length(TestWindow),1);
EWMA975 = zeros(length(TestWindow),1);
EWMA99 = zeros(length(TestWindow),1);
ES95 = zeros(length(TestWindow),1);
ES975 = zeros(length(TestWindow),1);
ES99 = zeros(length(TestWindow),1);
%% estimation
Lambda = 0.94; % default setting.
Sigma2     = zeros(length(Returns),1);
Sigma2(1)  = Returns(1)^2;
for i = 2 : (TestWindowStart-1)
    Sigma2(i) = (1-Lambda) * Returns(i-1)^2 + Lambda * Sigma2(i-1);
end

for t = TestWindow 
    k     = t - TestWindowStart + 1;
    Sigma2(t) = (1-Lambda) * Returns(t-1)^2 + Lambda * Sigma2(t-1);
    Sigma = sqrt(Sigma2(t));
    EWMA95(k) = -Zscore(1)*Sigma;
    ES95(k) =normpdf(Zscore(1))*Sigma/0.05;
    EWMA975(k) = -Zscore(2)*Sigma;
    ES975(k) =normpdf(Zscore(2))*Sigma/0.025;
    EWMA99(k) = -Zscore(3)*Sigma;
    ES99(k) =normpdf(Zscore(3))*Sigma/0.01;
end
%% Save the Results
data = [-EWMA95, -ES95, -EWMA975, -ES975, -EWMA99, -ES99];
T = array2table(data, 'VariableNames', {'Q95', 'E95','Q975', 'E975', 'Q99', 'E99'});
writetable(T, 'RM_SPX.xlsx');