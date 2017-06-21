%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% {PART 1b}                                                               %
% Purpose: Perform cross validation for RNN model selection               %
% Content: (i) Split into train and test sets for cross validation        %
% (ii) Train RNN                                                          %
% (iii) Predict on test set to compute Test MSEs for every fold iteration %
% (iv) Select the model with least Test MSE and return it                 %
% (v) Compute average fold MSE for all folds Test MSEs                    %
% Arguments: Input - fold no.,basic net model, input TS, target TS        %
% Output - best model, best model's parameters, avg Test MSE for all folds%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [bestNetFromCV,bestParamFromCV,AvgCrossValidationMSE] = CrossValidationRNN(K,Network,Input,Target)
rng(10);
% Input = timeSeries_Itrain';
% Target = timeSeries_Otrain';
Input = Input';
Target = Target';
% Network = net;
% K = 3;
indicesForCvFold = crossvalind('Kfold', size(Input,1), K);
for i = 1:K
        CVtestI = zeros(0,0);
        CVtestO = zeros(0,0);
        CVtrainI = zeros(0,0);
        CVtrainO = zeros(0,0);
       for n = 1:size(indicesForCvFold,1)
           if(mod(n,K) ~=0)
               CVtrainI = [CVtrainI;Input(n,:)];
               CVtrainO = [CVtrainO;Target(n,:)];
           else
               CVtestI=[CVtestI;Input(n,:)];
               CVtestO=[CVtestO;Target(n,:)];
           end  
       end
       CVtrainI = CVtrainI';
       CVtrainO = CVtrainO';
       
       [netCV{i}, trCV{i}] = train(Network,CVtrainI,CVtrainO);% Train on CV train set
       % Predict on test test created within Cross validation
       for j = 1 : size(CVtestI,1)
           predictedCVtest(j)=netCV{i}(CVtestI(j,:)');
       end
       avgFoldMSE(i) = mse(netCV{i}, CVtestO', predictedCVtest);
end
[minMSE, index] = min(avgFoldMSE);
bestNetFromCV = netCV(index);
bestParamFromCV  = trCV(index);
AvgCrossValidationMSE = mean(avgFoldMSE);
end

