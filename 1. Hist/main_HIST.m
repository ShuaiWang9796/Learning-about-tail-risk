clc
clear
%% data loading
data = xlsread('SPXR.csv');
Returns = data;
SampleSize = length(Returns);
%% Define the parameters
EstimationWindowSize = 252;
testlen = length(data)-floor(0.75*length(data));
%% Dividing the test set
TestWindowStart = SampleSize - testlen + 1;
TestWindowEnd = SampleSize;
TestWindow = TestWindowStart:TestWindowEnd;
ReturnsTest = Returns(TestWindow);
%% Forecast 95% VaR and ES 
VaRLevel = 0.95;
VaR_95 = zeros(length(TestWindow),1);
ES_95 = zeros(length(TestWindow),1);
for t = TestWindow
   i = t - TestWindowStart + 1;
   EstimationWindow = t-EstimationWindowSize:t-1;   
   [VaR_95(i),ES_95(i)] = hHistoricalVaRES(Returns(EstimationWindow),VaRLevel);  
end
% draw
figure;
plot(Returns( TestWindow))
hold on
plot(-VaR_95)
hold on
plot(-ES_95)
legend('Returns','VaR','ES','Location','southeast')
title('Historical VaR and ES')
grid on

%% Forecast 97.5% VaR and ES 
VaRLevel = 0.975;
VaR_975 = zeros(length(TestWindow),1);
ES_975 = zeros(length(TestWindow),1);
for t = TestWindow
   i = t - TestWindowStart + 1;
   EstimationWindow = t-EstimationWindowSize:t-1;   
   [VaR_975(i),ES_975(i)] = hHistoricalVaRES(Returns(EstimationWindow),VaRLevel);  
end
% draw
figure;
plot(Returns( TestWindow))
hold on
plot(-VaR_975)
hold on
plot(-ES_975)
legend('Returns','VaR','ES','Location','southeast')
title('Historical VaR and ES')
grid on

%% Forecast 99% VaR and ES 
VaRLevel = 0.99;
VaR_99 = zeros(length(TestWindow),1);
ES_99 = zeros(length(TestWindow),1);
for t = TestWindow
   i = t - TestWindowStart + 1;
   EstimationWindow = t-EstimationWindowSize:t-1;   
   [VaR_99(i),ES_99(i)] = hHistoricalVaRES(Returns(EstimationWindow),VaRLevel);  
end
% plot
figure;
plot(Returns( TestWindow))
hold on
plot(-VaR_99)
hold on
plot(-ES_99)
legend('Returns','VaR','ES','Location','southeast')
title('Historical VaR and ES')
grid on
%% save
data = [-VaR_95, -ES_95, -VaR_975, -ES_975, -VaR_99, -ES_99];
T = array2table(data, 'VariableNames', {'Q95', 'E95','Q975', 'E975', 'Q99', 'E99'});
writetable(T, 'hist_SPX.xlsx');
