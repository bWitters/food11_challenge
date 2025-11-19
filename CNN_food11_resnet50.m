close all
numClasses = numel(categories(imds.Labels));
net = resnet50;

net.Layers
inputSize = net.Layers(1).InputSize;
augmenter = imageDataAugmenter(...
    'RandXReflection', true, ...
    'RandRotation', [-10 10], ...
    'RandScale', [0.9 1.1], ...
    'RandXTranslation', [-5 5], ...
    'RandYTranslation', [-5 5], ...
    'RandXShear', [-5 5]);

augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    DataAugmentation=augmenter);

augimdsValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation);

lgraph = layerGraph(net);

newFCLayer = fullyConnectedLayer(numClasses, ...
    'Name','new_fc', ...
    'WeightLearnRateFactor',10, ...
    'BiasLearnRateFactor',10);

newClassLayer = classificationLayer('Name','new_classoutput');

lgraph = replaceLayer(lgraph, 'fc1000', newFCLayer);
lgraph = replaceLayer(lgraph, 'ClassificationLayer_fc1000', newClassLayer);

% layers = [
%     imageInputLayer(inputSize)
%     convolution2dLayer(5,20)
%     batchNormalizationLayer
%     reluLayer
%     maxPooling2dLayer(2, 'Stride', 2,'Name','maxpooling1')
%     convolution2dLayer(5,40,"padding",1,'Name','conv2')%padding dimension d'entree=dim sortie
%     batchNormalizationLayer
%     reluLayer('Name','relu2')
%     maxPooling2dLayer(2, 'Stride', 2,'Name','maxpooling2')
%     fullyConnectedLayer(numClasses,'Name','fullyconnected',"WeightsInitializer","narrow-normal")
%     softmaxLayer('Name','softmax')
%     classificationLayer('Name','classification') ];

miniBatchSize = 128;%可以改
initialLearnRate = 1e-4;%可以改
options = trainingOptions("adam", ...
    "L2Regularization",0.001,...
    InitialLearnRate=initialLearnRate, ...
    MiniBatchSize=miniBatchSize, ...
    MaxEpochs=4, ... % 可以根据需要增加
    Shuffle="every-epoch", ...
    ValidationData=augimdsValidation, ... % 使用验证集进行监控
    ValidationFrequency=5, ... % 每 5 次迭代运行一次验证
    Verbose=false, ...
    Plots="training-progress");

net = trainNetwork(augimdsTrain, lgraph, options);
inputSize = networkInputSize(net);


[YPred, scores] = classify(net, augimdsValidation);


YValidation = imdsValidation.Labels;


figure
confusionchart(YValidation, YPred);
title('Confusion Matrix on Validation Data');

classNames = categories(YValidation);
numClasses = numel(classNames);


precision = zeros(1, numClasses);
recall = zeros(1, numClasses);
f1Score = zeros(1, numClasses);


for i = 1:numClasses
    
    currentClass = classNames{i};
    
    TP = sum(YValidation == currentClass & YPred == currentClass);
    
    FP = sum(YValidation ~= currentClass & YPred == currentClass);
    
    FN = sum(YValidation == currentClass & YPred ~= currentClass);
    
    precision(i) = TP / (TP + FP);
    
    recall(i) = TP / (TP + FN);
    
    f1Score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
end

disp('--- 类别级 F1-Score 结果 ---');
T = table(classNames', precision', recall', f1Score', ...
    'VariableNames', {'Class', 'Precision', 'Recall', 'F1_Score'});
disp(T);

macroF1Score = mean(f1Score);
fprintf('\nMacro-Averaged F1-Score: %.4f\n', macroF1Score);