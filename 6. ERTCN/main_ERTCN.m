clc; clear; close all;
%% data
Y = readtable("SPXR.csv");
Y = table2array(Y);
% feaures and target
lag = 3;
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
[inputn,inputps]=mapminmax(X_train', 0, 1);
[outputn,outputps]=mapminmax(Y_train', 0, 1);
inputn_test = mapminmax('apply',X_test',inputps); 
%% construct the QRTCN network
tau = [0.018, 0.0066, 0.0026];  % required probability level (expectile)
for i = 1:3
alpha = tau(i);
% parameter settings
numFilters = 32;         
filterSize = 2;         
dropoutFactor = 0.1;   
numBlocks = 1;           
numFeatures = height(inputn);      
outputFeatures=1;     

layers = TCNNet(numFilters,filterSize,dropoutFactor,numBlocks,numFeatures,outputFeatures, alpha);
options = trainingOptions('adam', ...      % Adam
    'MaxEpochs', 3000, ...                 % 
    'MiniBatchSize',128,...                 % 
    'InitialLearnRate', 1e-2, ...          % 
    'Shuffle', 'every-epoch', ...          % 
    'Verbose', 1);                         %
   %% Train the ERTCN network
   net = trainNetwork(inputn, outputn, layers, options);	
   %% In-sample forecasting (for Add and Mult processes)
   YPred1 = predict(net,inputn,'MiniBatchSize',1);
   PRE11 = mapminmax('reverse',YPred1, outputps);
   PRE1(:,i) = PRE11'; 
   %% Out-of-sample forecasting
   YPred2 = predict(net,inputn_test,'MiniBatchSize',1);
   YPred2 = mapminmax('reverse',YPred2, outputps);
   PRE2(:,i) = YPred2';
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

%% predict 97.5% ES
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

%% predict 99% ES
Q001 = PRE2(:,3);
for i = 1:length(Q001)
    ES001(i,:) = (1 + tau(3)/((1-2*tau(3))*0.01))*Q001(i);
end
% draw
hold on
plot(Q001)
plot(ES001)
plot(Y2)
hold off
legend('Q', 'ES', 'actual')

%% save the results
data = [Q005,ES005,Q0025,ES0025,Q001,ES001];
T = array2table(data, 'VariableNames', {'Q95', 'E95','Q975', 'E975', 'Q99', 'E99'});
writetable(T, 'ExpectileTCN_SPX.xlsx');