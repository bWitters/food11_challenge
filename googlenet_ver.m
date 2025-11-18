%% food11_googlenet_optimized.m
% 优化版（GoogLeNet）：增强 + 类均衡 + 冻结大部分层 + LR schedule + JSON 输出 + GPU 能耗监控

clear; clc; close all;

%% --------- 用户可修改的路径与参数 ----------
baseFolder = 'D:\4TCS1\TIP\food-11\food-11';   % <- 修改为你的路径
trainFolder = fullfile(baseFolder, 'train');
testFolder  = fullfile(baseFolder, 'test');

% 超参（可微调）
initialLearnRate = 1e-4;
miniBatchSize = 64;
maxEpochs = 10;
valPatience = 3;
learnRateDropFactor = 0.2;
learnRateDropPeriod = 5;

% GPU 监控开关
doGPUlog = true;      
gpuLogFile = fullfile(pwd,'gpu_power_log.txt');

%% --------- GPU 路径自动加入（保证 nvidia-smi 可用） ----------
if doGPUlog
    setenv('PATH', [getenv('PATH') ';C:\Windows\System32;C:\Program Files\NVIDIA Corporation\NVSMI']);
end

%% --------- 路径检查 ----------
if ~isfolder(trainFolder)
    error('训练文件夹不存在: %s', trainFolder);
end

%% --------- 1. 读取数据 ----------
imds = imageDatastore(trainFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
[imdsTrain, imdsVal] = splitEachLabel(imds, 0.7, 0.3, 'randomized');
classNames = categories(imdsTrain.Labels);
numClasses = numel(classNames);
fprintf('Train: %d images, Val: %d images, Classes: %d\n', numel(imdsTrain.Files), numel(imdsVal.Files), numClasses);

%% --------- 2. oversample ----------
tbl = countEachLabel(imdsTrain);
maxCount = max(tbl.Count);

if maxCount / min(tbl.Count) > 1.2
    fprintf('Performing simple oversampling to balance classes...\n');
    filesBalanced = {};
    labelsBalanced = {};
    for i = 1:height(tbl)
        lbl = tbl.Label(i);
        files = imdsTrain.Files(imdsTrain.Labels == lbl);
        rep = ceil(maxCount / numel(files));
        filesRep = repmat(files, 1, rep);
        filesRep = filesRep(1:maxCount);
        filesBalanced = [filesBalanced; filesRep(:)];
        labelsBalanced = [labelsBalanced; repmat(cellstr(lbl), numel(filesRep),1)];
    end
    imdsTrain = imageDatastore(filesBalanced);
    imdsTrain.Labels = categorical(labelsBalanced);
    fprintf('Balanced training set size: %d\n', numel(imdsTrain.Files));
else
    fprintf('Training set relatively balanced — skipping oversample step.\n');
end

%% --------- 3. 网络 ----------
net = googlenet;
inputSize = net.Layers(1).InputSize(1:2);

%% --------- 4. 数据增强 ----------
augmenter = imageDataAugmenter( ...
    'RandXReflection', true, ...
    'RandRotation', [-10 10], ...
    'RandScale', [0.9 1.1], ...
    'RandXTranslation', [-10 10], ...
    'RandYTranslation', [-10 10]);

augImdsTrain = augmentedImageDatastore(inputSize, imdsTrain, 'DataAugmentation', augmenter);
augImdsVal   = augmentedImageDatastore(inputSize, imdsVal);

%% --------- 5. 替换最后层 ----------
lgraph = layerGraph(net);
newLayers = [
    fullyConnectedLayer(numClasses, 'Name','new_fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    softmaxLayer('Name','new_softmax')
    classificationLayer('Name','new_classoutput')];

lgraph = replaceLayer(lgraph, 'loss3-classifier', newLayers(1));
lgraph = replaceLayer(lgraph, 'prob', newLayers(2));
lgraph = replaceLayer(lgraph, 'output', newLayers(3));

layers = lgraph.Layers;
connections = lgraph.Connections;

for i = 1:numel(layers)
    if strcmp(layers(i).Name, 'new_fc')
        layers(i).WeightLearnRateFactor = 10;
        layers(i).BiasLearnRateFactor = 10;
    end
end

lgraph = createLgraphUsingConnections(layers, connections);

%% --------- 6. 训练选项 ----------
options = trainingOptions('adam', ...
    'InitialLearnRate', initialLearnRate, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',learnRateDropFactor, ...
    'LearnRateDropPeriod',learnRateDropPeriod, ...
    'MiniBatchSize', miniBatchSize, ...
    'MaxEpochs',maxEpochs, ...
    'ValidationData',augImdsVal, ...
    'ValidationFrequency', floor(numel(imdsTrain.Files)/miniBatchSize), ...
    'Verbose',true, ...
    'Plots','training-progress', ...
    'ValidationPatience',valPatience);

%% --------- 7. GPU 能耗监控：启动 nvidia-smi ----------
if doGPUlog
    if isfile(gpuLogFile), delete(gpuLogFile); end
    cmd = sprintf('start /MIN cmd /c "nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits --loop-ms=500 > %s"', gpuLogFile);
    system(cmd);
    fprintf('GPU power logging started → %s\n', gpuLogFile);
end

%% --------- 8. 训练 ----------
tic;
trainedNet = trainNetwork(augImdsTrain, lgraph, options);
trainTime = toc;
fprintf('Training done: %.2f s\n', trainTime);

%% --------- GPU logging 结束，读取能耗 ----------
if doGPUlog
    system('taskkill /F /IM nvidia-smi.exe >nul 2>nul');  % 结束采样进程
    pause(0.5);

    if isfile(gpuLogFile)
        p = readmatrix(gpuLogFile);
        avgPower = mean(p);
        energyWh = avgPower * (trainTime/3600);

        fprintf('\n====== GPU ENERGY REPORT ======\n');
        fprintf('Average GPU Power : %.2f W\n', avgPower);
        fprintf('Training Time     : %.2f s\n', trainTime);
        fprintf('Total Energy Used : %.4f Wh\n', energyWh);
        fprintf('================================\n\n');
    else
        warning('GPU log file missing — energy unavailable.');
    end
end

%% --------- 9. 验证 ----------
[YPred, ~] = classify(trainedNet, augImdsVal);
YVal = imdsVal.Labels;
acc = mean(YPred == YVal);
fprintf('Validation accuracy: %.2f%%\n', acc*100);

figure; confusionchart(YVal, YPred); title('Confusion Matrix (Validation)');

%% --------- Helper ----------
function lgraph = createLgraphUsingConnections(layers, connections)
    lgraph = layerGraph();
    for i = 1:numel(layers)
        lgraph = addLayers(lgraph, layers(i));
    end
    for i = 1:size(connections,1)
        try
            lgraph = connectLayers(lgraph, connections.Source{i}, connections.Destination{i});
        catch
        end
    end
end
