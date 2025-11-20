clear; clc; close all;

baseFolder = 'D:\4TCS1\TIP\food-11\food-11';

trainFolder = fullfile(baseFolder, 'train');
testFolder  = fullfile(baseFolder, 'test');

imds = imageDatastore(trainFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.7, 0.3, 'randomized');
fprintf('Training images: %d, Validation images: %d\n', numel(imdsTrain.Files), numel(imdsValidation.Files));
classNames = categories(imdsTrain.Labels);
numClasses = numel(classNames);
fprintf('Number of classes: %d\n', numClasses);

try
    net = resnet50;
    netName = 'resnet50';
catch
    warning('resnet50 not available. Using googlenet...');
    net = googlenet;
    netName = 'googlenet';
end
fprintf('Using %s\n', netName);

inputSize = net.Layers(1).InputSize(1:2);

augTrain = imageDataAugmenter('RandRotation',[-10,10], 'RandXTranslation',[-5 5], 'RandYTranslation',[-5 5]);
augImdsTrain = augmentedImageDatastore(inputSize, imdsTrain, 'DataAugmentation', augTrain);
augImdsVal   = augmentedImageDatastore(inputSize, imdsValidation);

if strcmp(netName,'resnet50')
    lgraph = layerGraph(net);
    lgraph = removeLayers(lgraph, {'fc1000','fc1000_softmax','ClassificationLayer_fc1000'});
    newLayers = [
        fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classoutput')];
    lgraph = addLayers(lgraph,newLayers);
    lgraph = connectLayers(lgraph,'avg_pool','fc');
else
    lgraph = layerGraph(net);
    lgraph = removeLayers(lgraph, {'loss3-classifier','prob','output'});
    newLayers = [
        fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classoutput')];
    lgraph = addLayers(lgraph,newLayers);
    lgraph = connectLayers(lgraph,'pool5-drop_7x7_s1','fc');
end

options = trainingOptions('adam', ...
    'MiniBatchSize',64, ...
    'MaxEpochs',10, ...
    'InitialLearnRate',1e-4, ...
    'ValidationData',augImdsVal, ...
    'ValidationFrequency',50, ...
    'Verbose',true, ...
    'Plots','training-progress', ...
    'ValidationPatience',1);

sampleInterval = 500;
gpuPowerLog = fullfile(pwd,'gpu_power_log.txt');

system(sprintf('start /B nvidia-smi --loop-ms=%d --query-gpu=power.draw --format=csv,noheader,nounits > "%s"', sampleInterval, gpuPowerLog));

tic
trainedNet = trainNetwork(augImdsTrain, lgraph, options);
trainTime = toc;
fprintf('Training finished in %.2f seconds.\n', trainTime);

system('taskkill /F /IM nvidia-smi.exe');

if isfile(gpuPowerLog)
    powerData = readmatrix(gpuPowerLog);
    avgPower = mean(powerData);
    energyWh = avgPower * (trainTime/3600);
    fprintf('Average GPU Power: %.2f W\n', avgPower);
    fprintf('Estimated GPU Energy Consumption: %.4f Wh\n', energyWh);
else
    warning('GPU power log not found. Energy not computed.');
end

[YPred, ~] = classify(trainedNet, augImdsVal);
YValidation = imdsValidation.Labels;
accuracy = mean(YPred == YValidation);
fprintf('Validation Accuracy: %.2f%%\n', accuracy*100);

figure;
plotconfusion(YValidation, YPred);
title('Confusion Matrix (Validation Set)');

imdsTest = imageDatastore(testFolder, 'IncludeSubfolders', false);
augImdsTest = augmentedImageDatastore(inputSize, imdsTest);
predLabels = classify(trainedNet, augImdsTest);

results = struct();
for i = 1:numel(imdsTest.Files)
    [~, name, ~] = fileparts(imdsTest.Files{i});
    results.(name) = char(predLabels(i));
end

jsonOutputFile = fullfile(pwd,'predictions_food11.json');
jsonText = jsonencode(results);
fid = fopen(jsonOutputFile,'w');
if fid == -1
    error('Cannot open file %s for writing.', jsonOutputFile);
end
fprintf(fid,'%s',jsonText);
fclose(fid);
fprintf('✅ Prediction JSON saved to: %s\n', jsonOutputFile);
