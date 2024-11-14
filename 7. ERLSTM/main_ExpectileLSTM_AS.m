clc; clear; close all;
%% data
Y = readtable("SPXR.csv");
Y = table2array(Y);
% feaures and target
lag = 2;
for i = 1:length(Y)-lag
    feature(i,:) = Y(i:i+lag-1); 
    target(i,:) = Y(i+lag);
end
% Test set length
testlen = length(Y)-floor(0.75*length(Y));
% Divide the training set and test set
X1 = feature(1:length(feature)-testlen, :);
Y1 = target(1:length(target)-testlen, :);
X2 = feature(length(feature)-testlen+1:end, :);
Y2 = target(length(target)-testlen+1:end, :);
% Consider the asymmetry
X_train = [max(X1,0),-min(X1,0)];
Y_train = Y1;
X_test = [max(X2,0),-min(X2,0)];
Y_test = Y2;
%% Normalization
[inputn,inputps]=mapstd(X_train',mean(mean(X_train)),mean(std(X_train)));
[outputn,outputps]=mapstd(Y_train', mean(Y_train), std(Y_train));
inputn_test = mapstd('apply',X_test',inputps); 
%% construct the ERLSTM network
tau = [0.0175, 0.0076, 0.003];  
for i = 1:3
alpha = tau(i);
% parameter settings
numResponses = 1;
featureDimension = size(inputn,1);
numHiddenUnits1 = 64;
numHiddenUnits2 = 32;
layers = [ ...
    sequenceInputLayer(featureDimension)
    lstmLayer(numHiddenUnits1,'OutputMode','sequence')
    dropoutLayer(0.2)
    lstmLayer(numHiddenUnits2,'OutputMode','sequence')
    dropoutLayer(0.2)
    fullyConnectedLayer(numResponses)
    ExpectileRegressionLayer('expRegLayer', alpha)];
options = trainingOptions('adam', ...
    'MaxEpochs', 200, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 0.01, ...
    'GradientThreshold', 1, ...
    'ExecutionEnvironment','gpu',...
    'Shuffle', 'every-epoch', ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 50);
   %% Train the ERLSTM network
   net = trainNetwork(inputn,outputn,layers,options);
   %% In-sample forecasting (for Add and Mult processes)
   YPred1 = predict(net,inputn,'MiniBatchSize',1);
   PRE11 = mapstd('reverse',YPred1, outputps);
   PRE1(:,i) = PRE11';
   sum(PRE1(:,i)<Y1)/length(PRE1(:,i))   
   %% Out-of-sample forecasting
   YPred2 = predict(net,inputn_test,'MiniBatchSize',1);
   YPred2 = mapstd('reverse',YPred2, outputps);
   PRE2(:,i) = YPred2';
   sum(PRE2(:,i)<Y2)/length(PRE2(:,i))
end

%% predict 95% ES (equation. 14)
Q005 = PRE2(:,1);
for i = 1:length(Q005)
    ES005(i,:) = (1 + tau(1)/((1-2*tau(1))*0.05))*Q005(i);
end
% 画图
hold on
plot(Q005)
plot(ES005)
plot(Y2)
hold off
legend('Q', 'ES', 'actual')
%% predict 97.5% ES (equation. 14)
Q0025 = PRE2(:,2);
for i = 1:length(Q0025)
    ES0025(i,:) = (1 + tau(2)/((1-2*tau(2))*0.025))*Q0025(i);
end
% 画图
hold on
plot(Q0025)
plot(ES0025)
plot(Y2)
hold off
legend('Q', 'ES', 'actual')
%% predict 99% ES (equation. 14)
Q001 = PRE2(:,3);
for i = 1:length(Q001)
    ES001(i,:) = (1 + tau(3)/((1-2*tau(3))*0.01))*Q001(i);
end
% 画图
hold on
plot(Q001)
plot(ES001)
plot(Y2)
hold off
legend('Q', 'ES', 'actual')

%% save the results
data = [Q005,ES005,Q0025,ES0025,Q001,ES001];
T = array2table(data, 'VariableNames', {'Q95', 'E95','Q975', 'E975', 'Q99', 'E99'});
writetable(T, 'ExpectileLSTM_SPX.xlsx');
