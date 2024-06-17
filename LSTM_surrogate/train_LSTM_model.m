clearvars

%% load results file

% results files contain two matrices, X and dVAF
% 
% - dVAF (num_experiments x num_years): the change in Volume above
% flotation for the Amundsen Sea region for each forward model experiment
% and each year, with respect to initial Volume above flotation at the
% start of the experiment, after initialisation
% - X (num_experiments x num_parameters): the set of uncertain model
% parameters and climate model ensemble ID

resultsFile = 'forwardSimulationResults';
load(resultsFile);

%% seperating data into training, validation and testing sets

numObservations = size(dVAF,1);

for ii = 1:numObservations
    targets{ii} = dVAF(ii,:);
    data{ii} = repmat(X(ii,:)',1,80);
end

idxTrain = 1:floor(0.8*numObservations);
idxVal = floor(0.8*numObservations)+1:floor(0.9*numObservations);
idxTest = floor(0.9*numObservations)+1:numObservations;
XTrain = data(idxTrain);
Xval = data(idxVal);
XTest = data(idxTest);
TTrain = targets(idxTrain);
TTest = targets(idxTest);
Tval = targets(idxVal);


muX = mean(cat(2,XTrain{:}),2);
sigmaX = std(cat(2,XTrain{:}),0,2);

muT = mean(cat(2,TTrain{:}),2);
sigmaT = std(cat(2,TTrain{:}),0,2);

for n = 1:numel(XTrain)
    XTrain{n} = (XTrain{n} - muX) ./ sigmaX;
    TTrain{n} = (TTrain{n} - muT) ./ sigmaT;
end

for n = 1:numel(XTest)
    XTest{n} = (XTest{n} - muX) ./ sigmaX;
    TTest{n} = (TTest{n} - muT) ./ sigmaT;
end

for n = 1:numel(Xval)
    Xval{n} = (Xval{n} - muX) ./ sigmaX;
    Tval{n} = (Tval{n} - muT) ./ sigmaT;
end

%% define and train network
numFeatures = size(XTrain{1},1);
numResponses = size(TTrain{1},1);
numHiddenUnits = 33;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    dropoutLayer(0.37)
    fullyConnectedLayer(numResponses)
    huberRegressionLayer('huber',0.75)];

maxEpochs = 1000;
miniBatchSize = 32;

options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'ValidationData',{Xval,Tval}, ...
    'InitialLearnRate',0.095, ...
    'LearnRateSchedule','piecewise', ... % Update the learning rate periodically by multiplying it by a drop factor
    'LearnRateDropPeriod',10,... % Number of epochs for dropping the learning rate
    'GradientThreshold',1, ... %%  If the gradient exceeds the value of GradientThreshold, then the gradient is clipped
    'Shuffle','every-epoch', ...
    'Plots','training-progress',...
    'L2Regularization',0.003, ...
    'ValidationPatience',10,...
    'Verbose',0);

net = trainNetwork(XTrain,TTrain,layers,options);

%% prediction

YPred = predict(net,XTest);
