clc; clear; close all;
%% data
Y = readtable("SPXR.csv");
Y = table2array(Y);
% feaures and target
lag = 1;
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
%% construct the QRLSTM network
tau = [0.05, 0.025, 0.01];
for i = 1:3
alpha = tau(i);
% parameter settings
numResponses = 1;
featureDimension = size(inputn,1);
numHiddenUnits1 = 100;
numHiddenUnits2 = 100;
layers = [ ...
    sequenceInputLayer(featureDimension)
    lstmLayer(numHiddenUnits1,'OutputMode','sequence')
    dropoutLayer(0.2)       
    %lstmLayer(numHiddenUnits2,'OutputMode','sequence')
    %dropoutLayer(0.25)    
    fullyConnectedLayer(numResponses)
    quanRegressionLayer('quanRegLayer', alpha)];
options = trainingOptions('adam', ...
    'MaxEpochs', 200, ...
    'MiniBatchSize', 128, ...
    'InitialLearnRate', 0.01, ...
    'GradientThreshold', 1, ...
    'ExecutionEnvironment','gpu',...
    'Shuffle', 'every-epoch', ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 100);
   %% Train the QRLSTM network
   net = trainNetwork(inputn,outputn,layers,options);
   %% In-sample forecasting (for Add and Mult processes)
   YPred1 = predict(net,inputn,'MiniBatchSize',1);
   YPred1 = mapstd('reverse',YPred1, outputps);
   PRE1(:,i) = YPred1';
   sum(PRE1(:,i)<Y1)/length(PRE1(:,i))   
   %% Out-of-sample forecasting
   YPred2 = predict(net,inputn_test,'MiniBatchSize',1);
   YPred2 = mapstd('reverse',YPred2, outputps);
   PRE2(:,i) = YPred2';
   sum(PRE2(:,i)<Y2)/length(PRE2(:,i))
end

%% Train the parameters of Add process to generate out-of-sample ES prediction results
for j = 1:3
Q = PRE1(:,j);
y = Y1;
func_to_minimize = @(x) double(mean(-log((tau(j)-1)./(Q-Add(x,Q,y))) - (y-Q).*(tau(j)-(y<=Q))./(tau(j)*(Q-Add(x,Q,y))) + y./(Q-Add(x,Q,y))));
initial_guess = [0.5, 0.5, 0.5];
% Constraints
LB = [0, 0, 0]; % lower bounds
UB = []; % no upper bounds
optimal_params = fmincon(func_to_minimize, initial_guess, [], [], [], [], LB, UB);
x = optimal_params;
z = 0;
for i = 2:length(Q)
 if y(i-1) <= Q(i-1)
    z(i,:) = x(1) + x(2)*(Q(i-1)-y(i-1)) + x(3)*z(i-1);
 else
    z(i,:) = z(i-1);
 end
end
% predict ES
z2 = z(end);
PRE22 = PRE2(:,j);
y = Y2;
for i = 2:length(PRE22)
 if y(i-1) <= PRE22(i-1)
    z2(i) = x(1) + x(2)*(PRE22(i-1)-y(i-1)) + x(3)*z2(i-1);
 else
    z2(i) = z2(i-1);
 end
end
z2 = z2';

ES2(:,j) = PRE22-z2;
% DRAW
figure
hold on
plot(PRE2(:,j))
plot(ES2(:,j))
plot(Y2)
hold off
legend('Q', 'ES', 'actual')
end

%% Train the parameters of Mult process to generate out-of-sample ES prediction results

for j = 1:3
Q = PRE1(:,j);
y = Y1;
func_to_minimize = @(x) double(mean(-log((tau(j)-1)./(Mult(x,Q))) - (y-Q).*(tau(j)-(y<=Q))./(tau(j)*(Mult(x,Q))) + y./(Mult(x,Q))));
initial_guess = 0;
LB = []; % no lower bounds
UB = []; % no upper bounds
optimal_params = fmincon(func_to_minimize, initial_guess, [], [], [], [], LB, UB);
x = optimal_params;
for i = 1:length(Q)
    z(i,:) = (1 + exp(x))*Q(i);
 end
ES = z;
% predict ES
z2 = z(end);
PRE222 = PRE2(:,j);
for i = 1:length(PRE222)
    z2(i,:) = (1 + exp(x))*PRE222(i);
end
ES3(:,j) = z2;

figure
hold on
plot(PRE2(:,j))
plot(ES3(:,j))
plot(Y2)
hold off
legend('Q', 'ES', 'actual')
end
%% Save the results
data = [PRE2(:,1),ES2(:,1),PRE2(:,2),ES2(:,2),PRE2(:,3),ES2(:,3)];
data2 = [PRE2(:,1),ES3(:,1),PRE2(:,2),ES3(:,2),PRE2(:,3),ES3(:,3)];
T = array2table(data, 'VariableNames', {'Q95', 'E95','Q975', 'E975', 'Q99', 'E99'});
T2 = array2table(data2, 'VariableNames', {'Q95', 'E95','Q975', 'E975', 'Q99', 'E99'});
writetable(T, 'QRLSTM_SPX_Add.xlsx');
writetable(T2, 'QRLSTM_SPX_Mult.xlsx');