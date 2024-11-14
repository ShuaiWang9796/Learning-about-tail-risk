clc; clear; close all;
%% data
Y = readtable("SPXR.csv");
Y = table2array(Y);
% feaures and target
lag = 4;
k = 10;
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
XTrain = [max(X1,0),-min(X1,0)]';
YTrain = Y1';
XTest = [max(X2,0),-min(X2,0)]';
Ytest = Y2;
layers = [ ...  
    sequenceInputLayer([height(XTrain),1,1],'name','input')   % input layer
    lstmLayer(50,'Outputmode','sequence','name','hidden1') 
    %dropoutLayer(0.1,'name','dropout_1')
    lstmLayer(50,'Outputmode','last','name','hidden1') 
    %dropoutLayer(0.1,'name','dropout_1')
    fullyConnectedLayer(50,'name','fullconnect')
    fullyConnectedLayer(k+1,'name','fullconnect') % fullyconnect layer (Consistent with the dimension of the parameter set)
    ];

% parameter settings
miniBatchSize = 128;
numEpochs = 60;
learnRate = 0.005;

numObservations = numel(YTrain);
numIterationsPerEpoch = floor(numObservations./miniBatchSize);
averageGrad = [];
averageSqGrad = [];
numIterations = numEpochs * numIterationsPerEpoch;
%monitor = trainingProgressMonitor(Metrics="Loss",Info="Epoch",XLabel="Iteration");
epoch = 0;
iteration = 0;
net = dlnetwork(layers);
while epoch < numEpochs % && ~monitor.Stop
    epoch = epoch + 1;
    idx=1:size(XTrain,2);
    XTrain = XTrain(1:height(XTrain),idx);
    YTrain = YTrain(1,idx);
    i = 0;
    while i < numIterationsPerEpoch %&& ~monitor.Stop
        i = i + 1;
        iteration = iteration + 1;
    switch iteration
      case numEpochs/2
        learnRate = learnRate *0.1;   
    end
        idx = (i - 1) * miniBatchSize+1 : i * miniBatchSize;
        X = XTrain(1:height(XTrain),idx);
        T = YTrain(1,idx);
        X = dlarray(X, 'SBCST');
        T = dlarray(T);
        [loss(epoch, i),gradients] = dlfeval(@Loss_W_QsplineC,net,X,T,k);
        [net,averageGrad,averageSqGrad] = adamupdate(net,gradients,averageGrad,averageSqGrad,iteration,learnRate);
    end
    fprintf('epoch=%d, loss= %.4f\n', epoch, sum(loss(epoch,:)));
end
%% parameter set → risk measures
XTrain = dlarray(XTrain, 'SBCST');
YTrain = predict(net, XTrain);
YTrain =  extractdata(YTrain);
beta0_Train = YTrain(1,:)';
gamma_Train = YTrain(2:k+1,:)';
gamma_Train = log(1+exp(gamma_Train));
beta0_Train = log(1+exp(beta0_Train));

% test set
XTest = dlarray(XTest, 'SBCST');
YTest = predict(net, XTest);
YTest =  extractdata(YTest);
beta0 = YTest(1,:)';
gamma = YTest(2:k+1,:)';
gamma = log(1+exp(gamma));
beta0 = log(1+exp(beta0));
% predict risk measures
[VaR005, ES005] = Qpredict_risk(beta0, gamma, 0.05, beta0_Train, gamma_Train, Y1, k);
quantileloss(0.05,  VaR005(252+1:end), Y2(252+1:end))
[VaR0025, ES0025] = Qpredict_risk(beta0, gamma, 0.025, beta0_Train, gamma_Train, Y1, k);
quantileloss(0.025,  VaR0025(252+1:end), Y2(252+1:end))
[VaR001, ES001] = Qpredict_risk(beta0, gamma, 0.01, beta0_Train, gamma_Train, Y1, k);
quantileloss(0.01,  VaR001(252+1:end), Y2(252+1:end))

%% save the results
data = [VaR005,ES005,VaR0025,ES0025,VaR001,ES001];
T = array2table(data, 'VariableNames', {'Q95', 'E95','Q975', 'E975', 'Q99', 'E99'});
writetable(T, 'LWCRPS_SPX.xlsx');



%% The above is the code of LWCRPS-LSTM model, and the code of LWCRPS-GRU and LWCRPS-TCN 
%% can be constructed by referring to ERGRU and ERTCN models in this project.