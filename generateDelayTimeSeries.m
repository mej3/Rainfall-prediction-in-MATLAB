%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% {PART 1a}                                                                %
% Purpose: Generate input and target time series for RNN training         %
% Content: Apply sliding window to generate input-target data             %
% Arguments: Input - filename & path, sliding window size, sample:Yes/No  %
% Output - input time series and output time series                       %
% Note: sample:Yes/No determines if sampling needs to be done by avoiding %
% overlapping time series sequence                                        %
% Author: Mithu James, student MSc Data Science, CityUniversity of London %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Create timeSeries
function [inputTS targetTS] = generateDelayTimeSeries(filePathName,slidingWindow,subSample)
    weather = xlsread(filePathName);
    mainTS = weather(:,1);
    noOfDatapts = length(mainTS);
    slidingWindowWidthNTarget = slidingWindow + 1;
    timeSeries = [];
    %For loop generates multiple sequence of observations as data points
    for i=1:(noOfDatapts-slidingWindow)   
        tsBeg = i;
        tsEnd = i+slidingWindowWidthNTarget-1;
        if(tsEnd <= noOfDatapts)
            tempTS = mainTS(tsBeg:tsEnd);
        else
            noOfPtsToFill = tsEnd - noOfDatapts;
            tempTS = mainTS(tsBeg:noOfDatapts);
            tempTS(end+1:end+noOfPtsToFill) = mainTS(1:noOfPtsToFill);
        end
        timeSeries(i,:) = tempTS;
    end
    inputTS = timeSeries(:,1:slidingWindow)';
    if(subSample ==1)% samples to avoid sequence already part of another
        inputTS = inputTS(:,1:slidingWindow:end);
    end
    targetTS = inputTS(:,2:end);
    inputTS = inputTS(:,1:(end-1));
end
