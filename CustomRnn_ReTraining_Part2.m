%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% {PART 2}                                                                %
% Purpose: Retraining of RNN                                              %
% Content: (i) Generate train and test sets                               %
% (ii) Train with K-fold cross validation                                 %
% (iii) Compute MSE for  model trained on entire training set             %
% (iv) Compute test set error                                             %
% (v) Retrieve weights and biases for final model                         %
% Team: Mithu James & Shagun Khare, City, University of London              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initial parameters
% rng(100);
% tic;
slidingWindowSize = 2;neuronsH1 = 3;activationFn = 2;trainAlgo = 4;%CHANGE FOR RE-TRAINING
% Selecting activation function
if(activationFn == 1)
    activationFunction = 'logsig';
elseif(activationFn == 2)
    activationFunction = 'tansig';
end
% Selecting training algorithm
if(trainAlgo == 1)
    trainingAlgorithm = 'trainlm';
elseif(trainAlgo == 2)
    trainingAlgorithm = 'traingdm';
elseif(trainAlgo == 3)
    trainingAlgorithm = 'traingda';
elseif(trainAlgo == 4)
    trainingAlgorithm = 'traingdx';    
end
% Generating training timeseries
[I_train] = generateDelayTimeSeries('ukWeatherNormalTrain.xlsx',slidingWindowSize,1);
timeSeries_Itrain = (I_train(:,1:end-1))';
timeSeries_Otrain = (I_train(:,2:end))';
timeSeries_Itrain = tonndata(timeSeries_Itrain,false,false);
timeSeries_Otrain = tonndata(timeSeries_Otrain,false,false);
% Generating testing timeseries
[I_test T_test] = generateDelayTimeSeries('ukWeatherNormalTest.xlsx',slidingWindowSize,1);
timeSeries_Itest = (I_test(:,1))';
timeSeries_Otest = (T_test(:,1))';
timeSeries_Itest = tonndata(timeSeries_Itest,false,false);
timeSeries_Otest = tonndata(timeSeries_Otest,false,false);
%% Set final network parameters
selectedModelNET = mdlNET{6};%CHANGE FOR RE-TRAINING
selectedModelTR = mdlTR{6};%CHANGE FOR RE-TRAINING
neuronsH1 = selectedModelNET{1}.layers{1}.size;
activationFunction = selectedModelNET{1}.layers{1}.transferFcn;
outputFunction = selectedModelNET{1}.layers{2}.transferFcn;
trainingAlgorithm = selectedModelTR{1,1}.trainFcn;
regularisationL1 = selectedModelNET{1}.performParam.regularization;
epochs = selectedModelTR{1}.num_epochs + 100;
finalResult = {'Window', 'Neurons', 'Activation','Algorithm','SetEpochs','TrainMSE','TestMSE',...
    'Regularisation','MaxEpoch','BestEpoch'};
%% Create a neural tuneNet
hiddenSizes = [1];
trainFcn = trainingAlgorithm;
tuneNet = fitnet(hiddenSizes,trainFcn);
%% Configure parameters to modify as Recurrent NN
tuneNet.numInputs = 1;
tuneNet.inputs{1}.size = 1;
tuneNet.layers{1}.dimensions = 1;
tuneNet.numLayers = 2;
tuneNet.biasConnect = [0; 1];
tuneNet.inputConnect = [1; 0];
% tuneNet.layerConnect = [0 1; 1 0];%Jordan
% tuneNet.layerConnect = [1 1; 1 0];%ElmanJordan
tuneNet.layerConnect = [1 0; 1 0];%Elman
tuneNet.layers{1}.size = neuronsH1;tuneNet.layers{1}.transferFcn = activationFunction;tuneNet.layers{1}.initFcn = 'initnw';
tuneNet.layers{2}.transferFcn = outputFunction;tuneNet.layers{2}.initFcn = 'initnw';
tuneNet.layerWeights{2,2}.delays = 1;%Elman
tuneNet.layerWeights{1,1}.delays = 1;%Elman
% tuneNet.layerWeights{1,2}.delays = 1;%Jordan
tuneNet.divideMode = 'time';  % Divide up every sample
tuneNet.divideParam.trainRatio = 90/100;
tuneNet.divideParam.valRatio = 10/100;
tuneNet.divideParam.testRatio = 0;
tuneNet.performFcn = 'mse';
tuneNet.trainParam.show = 50;%tuneNet.trainParam.lr = learningRate(lr);
tuneNet.trainParam.epochs = epochs; tuneNet.trainParam.goal = 1e-3;
tuneNet.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
tuneNet.plotFcns = {'plotperform','plottrainstate','plotresponse', ... 
    'ploterrcorr', 'plotinerrcorr'};
tuneNet.performParam.regularization = regularisationL1;
tuneNet.trainFcn = trainingAlgorithm;% tuneNet.trainParam.showWindow = false;
tuneNet = setwb(tuneNet,wb);
[tunedNetCV, tunedTrCV] = train(tuneNet,timeSeries_Itrain,timeSeries_Otrain);% Train on CV train set
% view(tuneNet);
%% Predict and compute training error
for j = 1 : size(timeSeries_Itrain,2)
    predictedOutput = tunedNetCV(timeSeries_Itrain);
    error = gsubtract(predictedOutput,timeSeries_Otrain);
    MSE_Train(j) = mse(error);
end
avgTrainMSE_afterRetraining = mean(MSE_Train);
%% Predict and compute test error
actual = [];
expected = [];
for j = 1 : size(I_test,2)
    timeSeries_Itest = (I_test(:,j))';
    timeSeries_Otest = (T_test(:,j))';
    predictedOutput = tunedNetCV(timeSeries_Itest');
    error = gsubtract(predictedOutput,timeSeries_Otest);
    MSE_Test(j) = mse(error);
    actual = [actual; predictedOutput];
    expected = [expected;timeSeries_Otest(1,1);timeSeries_Otest(1,2)];
end
avgTestMSE_afterRetraining = mean(MSE_Test);
% figure;plot(expected(:,1));hold on;plot(actual(:,1),'--');
%% Performance Matrix update
perfTunedArray(1) = size(timeSeries_Itest,2);perfTunedArray(2) = neuronsH1;
perfTunedArray(3) = activationFn; perfTunedArray(4) = trainAlgo;
perfTunedArray(5) = epochs;perfTunedArray(6) = avgTrainMSE_afterRetraining;
perfTunedArray(7) = avgTestMSE_afterRetraining;perfTunedArray(8) = regularisationL1;
perfTunedArray(9) = tunedTrCV.num_epochs; perfTunedArray(10) = tunedTrCV.best_epoch;
finalResult = [finalResult;num2cell(perfTunedArray)];
%% Bias
wb = getwb(tunedNetCV);
[b,IW,LW] = separatewb(tunedNetCV,wb);
WB = num2cell(wb);
weightAndBias = sprintf('%s ' ,WB{:});
weightAndBias = strtrim(weightAndBias);
%% Clear variables
clear vars activationFunction af error hiddenSizes j regularisationL1;
clear vars slidingWindowSize sqError ta trainFcn trainingAlgorithm epochs;
clear vars MSE_Train MSE_Test neuronsH1 outputFunction perfTunedArray;
%% Time
% toc;