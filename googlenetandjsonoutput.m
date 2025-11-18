%% ================================
%   FOOD-11 Training + Evaluation
%   ResNet50 / GoogLeNet + GPU Power
%   Final Integrated Version
% ================================

clear; clc; close all;

%% ======= Folder Setup =======

baseFolder = 'D:\4TCS1\TIP\food-11\food-11';

trainFolder = fullfile(baseFolder, 'train');
testFolder  = fullfile(baseFolder, 'test');

imds = imageDatastore(trainFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

[imdsTrain, imdsVal] = splitEachLabel(imds, 0.7, 0.3, 'randomized');
classNames = categories(imdsTrain.Labels);
numClasses = numel(classNames);

fprintf('Training images: %d, Validation images: %d\n', numel(imdsTrain.Files), numel(imdsVal.Files));
fprintf('Number of classes: %d\n', numClasses);

%% ======= Load Network (ResNet50 → fallback GoogLeNet) =======

try
    net = googlenet;
    netName = 'googlenet';
end

fprintf('Using network: %s\n', netName);

inputSize = net.Layers(1).InputSize(1:2);

%% ======= Data Augmentation =======

augmenter = imageDataAugmenter( ...
    'RandRotation', [-10 10], ...
    'RandXTranslation', [-5 5], ...
    'RandYTranslation', [-5 5]);

augTrain = augmentedImageDatastore(inputSize, imdsTrain, 'DataAugmentation', augmenter);
augVal   = augmentedImageDatastore(inputSize, imdsVal);

%% ======= Modify Final Layers =======

lgraph = layerGraph(net);

if strcmp(netName, 'resnet50')
    % Remove original head
    removeList = {'fc1000','fc1000_softmax','ClassificationLayer_fc1000'};
    lgraph = removeLayers(lgraph, removeList);

    % New layers
    newLayers = [
        fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classoutput')
    ];

    lgraph = addLayers(lgraph, newLayers);
    lgraph = connectLayers(lgraph, 'avg_pool','fc');

else
    % GoogLeNet
    removeList = {'loss3-classifier','prob','output'};
    lgraph = removeLayers(lgraph, removeList);

    newLayers = [
        fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classoutput')
    ];

    lgraph = addLayers(lgraph, newLayers);
    lgraph = connectLayers(lgraph, 'pool5-drop_7x7_s1','fc');
end

%% ======= Training Options =======

options = trainingOptions('adam', ...
    'MiniBatchSize', 64, ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 1e-4, ...
    'ValidationData', augVal, ...
    'ValidationFrequency', 50, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ValidationPatience', 3);

%% ======= GPU Power Monitoring =======

gpuLog = fullfile(pwd, 'gpu_power_log.txt');
intervalMs = 500;

system(sprintf('start /B nvidia-smi --loop-ms=%d --query-gpu=power.draw --format=csv,noheader,nounits > "%s"', ...
    intervalMs, gpuLog));

%% ======= Train Network =======

tic
trainedNet = trainNetwork(augTrain, lgraph, options);
trainTime = toc;

fprintf('Training finished in %.2f seconds.\n', trainTime);

%% Stop Power Logging
system('taskkill /F /IM nvidia-smi.exe');

%% ======= GPU Power Analysis =======

if isfile(gpuLog)
    powerData = readmatrix(gpuLog);
    avgPower = mean(powerData);
    energyWh = avgPower * (trainTime/3600);

    fprintf('Average GPU Power: %.2f W\n', avgPower);
    fprintf('Estimated Energy Consumption: %.4f Wh\n', energyWh);
else
    warning('Power log file missing. No GPU energy computed.');
end

%% ======= Validation Accuracy =======

[YPred, ~] = classify(trainedNet, augVal);
YVal = imdsVal.Labels;

accuracy = mean(YPred == YVal);
fprintf('Validation Accuracy: %.2f%%\n', accuracy*100);

figure;
plotconfusion(YVal, YPred);
title('Confusion Matrix – Validation');

%% ======= F1-score Computation =======

precision = zeros(numClasses,1);
recall = zeros(numClasses,1);
f1Score = zeros(numClasses,1);

for i = 1:numClasses
    cls = classNames{i};

    TP = sum(YPred == cls & YVal == cls);
    FP = sum(YPred == cls & YVal ~= cls);
    FN = sum(YPred ~= cls & YVal == cls);

    precision(i) = TP / (TP + FP + eps);
    recall(i)    = TP / (TP + FN + eps);
    f1Score(i)   = 2 * precision(i) * recall(i) / (precision(i) + recall(i) + eps);
end

macroF1 = mean(f1Score);
fprintf('Macro F1-score: %.4f\n', macroF1);

%% ======= Generate Predictions for Test Set (JSON Format) =======

imdsTest = imageDatastore(testFolder, 'IncludeSubfolders', false);
augTest = augmentedImageDatastore(inputSize, imdsTest);

[YPredTest, ~] = classify(trainedNet, augTest);

% Extract file names
[~, testNames, ~] = cellfun(@fileparts, imdsTest.Files, 'UniformOutput', false);

% Use Map (struct not allowed for numeric keys)
jsonMap = containers.Map('KeyType','char','ValueType','char');

for i = 1:numel(testNames)
    jsonMap(testNames{i}) = char(YPredTest(i));
end

jsonStr = jsonencode(jsonMap);

outputFile = fullfile(pwd, 'predictions_food11.json');
fid = fopen(outputFile,'w');
if fid == -1
    error('Cannot write JSON file.');
end

fprintf(fid, '%s', jsonStr);
fclose(fid);

fprintf('✅ Predictions JSON saved to: %s\n', outputFile);
