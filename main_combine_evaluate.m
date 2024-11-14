clc
clear
%% probability level
tau = 0.01;  % tau=[0.05, 0.025, 0.01]
%% data loading
Return = xlsread('SPXR.csv');
Return = Return(floor(0.75*length(Return))+2:end);
%% Select the forecast series corresponding to the probability level
if tau == 0.05
    n1 = 1; n2 = 2;
elseif tau == 0.025
    n1 = 3; n2 = 4;
else
    n1 = 5; n2 = 6;
end
%% Import all forecast results, and each forecast is n*6. [Q95 ES95 Q975 ES975 Q99 ES99]
% 1. Historical Simulation (Hist)
hist = xlsread('1. Hist\hist_SPX.xlsx');
hist = hist(end-length(Return)+1:end, n1:n2);
% 2. RiskMetrics
RM = xlsread('2. RiskMetrics\RM_SPX.xlsx');
RM = RM(end-length(Return)+1:end, n1:n2);
% 3. QRTCN-Add
QRTCN_Add = xlsread('3. QRTCN\QRTCN_SPX_Add.xlsx');
QRTCN_Add = QRTCN_Add(end-length(Return)+1:end, n1:n2);
% 4. QRTCN-Mult
QRTCN_Mult = xlsread('3. QRTCN\QRTCN_SPX_Mult.xlsx');
QRTCN_Mult = QRTCN_Mult(end-length(Return)+1:end, n1:n2);
% 5. QRLSTM-Add
QRLSTM_Add = xlsread('4. QRLSTM\QRLSTM_SPX_Add.xlsx');
QRLSTM_Add = QRLSTM_Add(end-length(Return)+1:end, n1:n2);
% 6. QRLSTM-Mult
QRLSTM_Mult = xlsread('4. QRLSTM\QRLSTM_SPX_Mult.xlsx');
QRLSTM_Mult = QRLSTM_Mult(end-length(Return)+1:end, n1:n2);
% 7. QRGRU-Add
QRGRU_Add = xlsread('5. QRGRU\QRGRU_SPX_Add.xlsx');
QRGRU_Add = QRGRU_Add(end-length(Return)+1:end, n1:n2);
% 8. QRGRU-Mult
QRGRU_Mult = xlsread('5. QRGRU\QRGRU_SPX_Mult.xlsx');
QRGRU_Mult = QRGRU_Mult(end-length(Return)+1:end, n1:n2);
% 9. ERTCN
ERTCN = xlsread('6. ERTCN\ExpectileTCN_SPX.xlsx');
ERTCN = ERTCN(end-length(Return)+1:end, n1:n2);
% 10. ERLSTM
ERLSTM = xlsread('7. ERLSTM\ExpectileLSTM_SPX.xlsx');
ERLSTM = ERLSTM(end-length(Return)+1:end, n1:n2);
% 11. ERGRU
ERGRU = xlsread('8. ERGRU\ExpectileGRU_SPX.xlsx');
ERGRU = ERGRU(end-length(Return)+1:end, n1:n2);
% 12. LWCRPS
LWCRPS = xlsread('9. LWCRPS\LWCRPS_SPX.xlsx');
LWCRPS = LWCRPS(end-length(Return)+1:end, n1:n2);
%% All results
all = [hist, RM, QRTCN_Add, QRTCN_Mult, QRLSTM_Add, QRLSTM_Mult,  ...
    QRGRU_Add, QRGRU_Mult, ERTCN, ERLSTM, ERGRU, LWCRPS];
%% Relative score combining (Windwos=252)
windows = 252;
lamda = 0.5;
w = [];
for j = windows+1:length(Return)
  for i = 1:2:width(all)
     allscore(i) = exp(-lamda * ALscore(Return(j-windows:j-1,1), all(j-windows:j-1,i), all(j-windows:j-1,i+1), tau));
  end
  for i = 1:2:width(all)
     w(j-windows, i:i+1) = exp(-lamda * ALscore(Return(j-windows:j-1,1), all(j-windows:j-1,i), all(j-windows:j-1,i+1), tau))/sum(allscore);
  end
end
combineforecast1 = all(windows+1:length(Return),:) .* w;
Qcombine1 = sum(combineforecast1(:, 1:2:width(all)), 2);
EScombine1 = sum(combineforecast1(:, 2:2:width(all)), 2);
[ALc1, FZGc1, NZc1, ASc1] = ESscore(Return(windows+1:length(Return)), Qcombine1, EScombine1, tau);
%% Simple average
w1 = ones(length(w), width(w)) * (1/(width(w)/2));
combineforecast2 = all(windows+1:length(Return),:) .* w1;
Qcombine2 = sum(combineforecast2(:, 1:2:width(all)), 2);
EScombine2 = sum(combineforecast2(:, 2:2:width(all)), 2);
[ALc2, FZGc2, NZc2, ASc2] = ESscore(Return(windows+1:length(Return)), Qcombine2, EScombine2, tau);
%% Combining all with selection and shrinkage
wfinal = [];
lamdaL1 = 0.5; lamdaL2 = 0.5; lamdaL3 = 0.7;
for j = windows+1:length(Return)
    wtemp = ones(width(all)/2, 1) * 2/width(all);
    yL1 = Return(j-windows:j-1, 1);
    func_to_minimize = @(wtemp) double(mean(-log((tau-1)./sum(all(j-windows:j-1,2:2:end) * wtemp, 2)) - (yL1-sum(all(j-windows:j-1,1:2:end) * wtemp, 2)).*(tau-(yL1<=sum(all(j-windows:j-1,1:2:end) * wtemp, 2))) ./ (tau.*sum(all(j-windows:j-1,2:2:end) * wtemp, 2))) + lamdaL3 * sum(abs(wtemp-1/width(wtemp))) + (1-lamdaL3) * sum(abs(wtemp-1/width(wtemp)).^2)); % 
    %func_to_minimize = @(wtemp) double(mean(-log((tau-1)./sum(all(j-windows:j-1,2:2:end) * wtemp, 2)) - (yL1-sum(all(j-windows:j-1,1:2:end) * wtemp, 2)).*(tau-(yL1<=sum(all(j-windows:j-1,1:2:end) * wtemp, 2))) ./ (tau.*sum(all(j-windows:j-1,2:2:end) * wtemp, 2))) + lamdaL1 * sum(abs(wtemp)) + (1-lamdaL2) * sum(abs(wtemp-1/width(wtemp)))); 
    Aeq = ones(1, numel(wtemp));
    beq = 1;
    nonlcon = [];
    lb = zeros(width(all)/2, 1);
    ub = ones(width(all)/2, 1);
    options = optimoptions('fmincon', 'Algorithm', 'interior-point');
    wfinal(:,j-windows) = fmincon(func_to_minimize, wtemp, [], [], Aeq, beq, lb, ub, nonlcon, options);
end
wfinal = wfinal';
QcombineL3 = sum(all(windows+1:length(Return), 1:2:width(all)) .* wfinal, 2);
EScombineL3 = sum(all(windows+1:length(Return), 2:2:width(all)) .* wfinal, 2);
[ALcL3, FZGcL3, NZcL3, AScL3] = ESscore(Return(windows+1:length(Return)), QcombineL3, EScombineL3, tau);
%% joint scores (AL, FZG, NZ, AS)
h = 1:2:width(all);
for i = 1:width(all)/2
 [AL(i), FZG(i), NZ(i), AS(i)] = ESscore(Return(windows+1:length(Return)), all(windows+1:length(Return),h(i)), all(windows+1:length(Return),h(i)+1), tau);
end
jointScore = [AL; FZG; NZ; AS];
jointScore = [jointScore, [ALc1, FZGc1, NZc1, ASc1]', [ALc2, FZGc2, NZc2, ASc2]', [ALcL3, FZGcL3, NZcL3, AScL3]'];
%% quantile score (QL)
QS = QL(Return(windows+1:length(Return)), all(windows+1:length(Return), 1:2:width(all)), tau);
QS1 = QL(Return(windows+1:length(Return)), Qcombine1, tau);
QS2 = QL(Return(windows+1:length(Return)), Qcombine2, tau);
QSL3 = QL(Return(windows+1:length(Return)), QcombineL3, tau);
QuantileScore = [QS, QS1, QS2, QSL3];
%
allScore = [QuantileScore; jointScore];
%% skill score
SkillScore=[];
for i = 1:width(allScore)
    SkillScore(:,i) = (1 - allScore(:,i)./allScore(:,1))*100;
end

%% Skill score （Corresponding to the results in Tables 6-8.）
SkillScore = SkillScore';