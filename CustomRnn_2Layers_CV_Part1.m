%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% {PART 1 - 1 hidden layer}                                                                %
% Purpose: Timeseries prediction with different recurrent neural networks %
% Content: (i) Generate train and test sets                               %
% (ii) Train with K-fold cross validation                                 %
% (iii) Provide MSEs for all trained models for model selection           %
% (iv) Compute train error on entire train set & hold-out test set error  %
% (v) Retrieve weights and biases forselected model                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Variable initialisation
clear all;close all;clc;
tic;%rng(100);
% Each of the arrays below has its iterator set to 1 to initialise with
% first array value when that array variable is excluded from grid search
swz = 1;slidingWindowSize = [2, 3];
nh1 = 1; neuronsH1 = [3, 4, 5, 6, 8];
af = 1; activationFn = [1,2];
ta = 1; trainAlgo = [2, 3,4];
lr = 1;learningRate = [0.01,0.1, 0.5,0.9];
m=1; momentum = [0.5,0.3,0.1,0.5];
r = 1; regularisationL1 = [0, 0.5, 1];
e = 1;epochs = [500, 200, 300];
KFold = 10;
c = 1;
gridSearchResult = {'Window', 'Neurons', 'Activation','Algorithm','LR',...
    'Momentum','Epochs','AvgCV MSE','TestMSE',...
    'BestTrainMSE','BestValidnMSE','Regularisation','MaxEpoch','BestEpoch'};
%% Grid search - enable each FOR loop as desired
for swz = 1:size(slidingWindowSize,2)
for nh1 = 1:size(neuronsH1,2)
for af = 1:size(activationFn,2)
if(activationFn(af) == 1)
    activationFunction = 'logsig';
elseif(activationFn(af) == 2)
    activationFunction = 'tansig';
end
for ta = 1:size(trainAlgo,2)
if(trainAlgo(ta) == 1)
    trainingAlgorithm = 'trainlm';
elseif(trainAlgo(ta) == 2)
    trainingAlgorithm = 'traingdm';
elseif(trainAlgo(ta) == 3)
    trainingAlgorithm = 'traingda';
elseif(trainAlgo(ta) == 4)
    trainingAlgorithm = 'traingdx';    
end
% for lr = 1:size(learningRate,2)
% for m = 1:size(momentum,2)
% for r = 1:size(regularisationL1,2)
% for e = 1:size(epochs,2)
    %% Generate training and test timeSeries
    [I_train] = generateDelayTimeSeries('ukWeatherNormalTrain.xlsx',slidingWindowSize(swz),1);
    timeSeries_Itrain = (I_train(:,1:end-1))';
    timeSeries_Otrain = (I_train(:,2:end))';
    timeSeries_Itrain = tonndata(timeSeries_Itrain,false,false);
    timeSeries_Otrain = tonndata(timeSeries_Otrain,false,false);

    [I_test T_test] = generateDelayTimeSeries('ukWeatherNormalTest.xlsx',slidingWindowSize(swz),1);
    timeSeries_Itest = (I_test(:,1))';
    timeSeries_Otest = (T_test(:,1))';
    timeSeries_Itest = tonndata(timeSeries_Itest,false,false);
    timeSeries_Otest = tonndata(timeSeries_Otest,false,false);
    %% Create a bare neural net
    hiddenSizes = [1];
    trainFcn = trainingAlgorithm;
    net = fitnet(hiddenSizes,trainFcn);
    %% Configure parameters to modify as Recurrent NN
    net.numInputs = 1;
    net.inputs{1}.size = 1;
    net.layers{1}.dimensions = 1;
    net.numLayers = 2;
    net.biasConnect = [0; 1];
    net.inputConnect = [1; 0];
    % Select one of next 3 lines to enable the desired network - Jordan, Elman etc.
    % The layerConnect was modified to include more hidden layers
    % net.layerConnect = [0 1; 1 0];%Jordan
    % net.layerConnect = [1 1; 1 0];%ElmanJordan
    net.layerConnect = [1 0; 1 0];%Elman
    net.layers{1}.size = neuronsH1(nh1);net.layers{1}.transferFcn = activationFunction;net.layers{1}.initFcn = 'initnw';
    net.layers{2}.transferFcn = 'purelin';net.layers{2}.initFcn = 'initnw';
    net.IW{1,1}(1:neuronsH1(nh1)) = 0.1;
    net.layerWeights{2,2}.delays = 1;%Elman
    net.layerWeights{1,1}.delays = 1;%Elman
    % net.layerWeights{1,2}.delays = 1;%Jordan
    net.divideFcn = 'dividerand';  % Divide data randomly
    net.divideMode = 'time';  % Divide up every sample
    net.divideParam.trainRatio = 90/100;
    net.divideParam.valRatio = 10/100;
    net.divideParam.testRatio = 0;
    net.performFcn = 'mse';
    net.trainParam.mc = momentum(m);
    net.trainParam.show = 50;net.trainParam.lr = learningRate(lr);
    net.trainParam.epochs = epochs(e); net.trainParam.goal = 1e-3;
    net.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
    net.plotFcns = {'plotperform','plottrainstate','plotresponse', ... 
        'ploterrcorr', 'plotinerrcorr'};
    net.performParam.regularization = regularisationL1(r);
    net.trainFcn = trainingAlgorithm; net.trainParam.showWindow = false;
    % view(net);
    %% Train custom neural network with recurrence in hidden layer
    [netCV, trCV, AvgFoldMSE] = CrossValidationRNNcv(KFold,net,timeSeries_Itrain,timeSeries_Otrain);
    mdlNET{c} = netCV; mdlTR{c} = trCV; mdlCVfoldMSE{c} = AvgFoldMSE;
    MSE_bestTrain = trCV{1}.best_perf;
    MSE_bestValidn = trCV{1}.best_vperf;
    MSE_stopReason = trCV{1}.stop;
    totalEpochs = trCV{1}.num_epochs;
    bestEpoch = trCV{1}.best_epoch;
    stopReason = trCV{1}.stop;
    disp(stopReason);
    %% Predict and compute training error
    for j = 1 : size(I_test,2)
        predictedOutput = netCV{1}(timeSeries_Itrain);
        error = gsubtract(predictedOutput,timeSeries_Otrain);
        MSE_Train(j) = mse(error);
    end
    avgTrainCvMSE_beforeRetraining = mean(MSE_Train);
    %% Predict and compute test error
    for j = 1 : size(I_test,2)
        timeSeries_Itest = (I_test(:,j))';
        timeSeries_Otest = (T_test(:,j))';
        predictedOutput = netCV{1}(timeSeries_Itest');
        error = gsubtract(predictedOutput,timeSeries_Otest);
        MSE_Test(j) = mse(error);
    end
    avgTestMSE_beforeRetraining = mean(MSE_Test);%saves mean MSE for every grid search
    SqError = 0;
    %% Retrieves the Network weight and bias for every model (not saved)
    wb = getwb(netCV{1});
    [b,IW,LW] = separatewb(netCV{1},wb);
    WB = num2cell(wb);
    weightAndBias = sprintf('%s ' ,WB{:});
    weightAndBias = strtrim(weightAndBias);
    %% Stores the result of every iteration during grid search
    perfArray(1) = slidingWindowSize(swz);perfArray(2) = neuronsH1(nh1);
    perfArray(3) = activationFn(af); perfArray(4) = trainAlgo(ta);
    perfArray(5) = learningRate(lr);perfArray(6) = momentum(m);
    perfArray(7) = epochs(e);perfArray(8) = AvgFoldMSE;perfArray(9) = avgTestMSE_beforeRetraining;
    perfArray(10) = MSE_bestTrain;perfArray(11) = MSE_bestValidn;
    perfArray(12) = regularisationL1(r);
    perfArray(13) = totalEpochs; perfArray(14) = bestEpoch;
    gridSearchResult = [gridSearchResult;num2cell(perfArray)];
    c = c+1;
% end
% end
% end
% end
end
end
end
end
%% Clear variables
clear vars activationFn activationFunction af b e epochs hiddenSizes IW j;
clear vars learningRate lr LW m momentum MSE_Test MSE_Train net neuronsH1 nh1;
clear vars predictedOutput r regularisationL1 sqError swz KFold perfArray;
clear vars ta trainAlgo trainFcn WB weightAndBias trainingAlgorithm;
clear vars totalEpochs bestEpoch MSE_bestTrain MSE_bestValidn MSE_stopReason;
toc;
