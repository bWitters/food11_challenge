close all


testFolder = "C:\Users\moqian\Desktop\food-11\test";
% ResNet-18 transfer learning
net = resnet18;
inputSize = net.Layers(1).InputSize;

% augmenter data

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

% remplace les deux derniers layers

lgraph = layerGraph(net);
newFCLayer = fullyConnectedLayer(numClasses, ...
    'Name','new_fc', ...
    'WeightLearnRateFactor',10, ...
    'BiasLearnRateFactor',10);
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph, 'fc1000', newFCLayer);
lgraph = replaceLayer(lgraph, 'ClassificationLayer_predictions', newClassLayer);



miniBatchSize = 128;
initialLearnRate = 1e-4;

options = trainingOptions("adam", ...
    "L2Regularization",0.1,...
    InitialLearnRate=initialLearnRate, ...
    MiniBatchSize=miniBatchSize, ...
    MaxEpochs=20, ... 
    Shuffle="every-epoch", ...
    ValidationData=augimdsValidation, ... 
    ValidationFrequency=5, ... 
    Verbose=false, ...
    Plots="training-progress");


net = trainNetwork(augimdsTrain, lgraph, options);

% Calculer f1-score,evaluer le reseau
[YPred, scores] = classify(net, augimdsValidation);
YValidation = imdsValidation.Labels;

figure;
confusionchart(YValidation, YPred);
title('Confusion Matrix on Validation Data');

precision = zeros(1, numClasses);
recall = zeros(1, numClasses);
f1Score = zeros(1, numClasses);

disp('Evaluation sur test data');
fprintf('%-20s %10s %10s %10s\n', 'Class', 'Precision', 'Recall', 'F1_Score');
fprintf('%-20s %10s %10s %10s\n', repmat('-', 1, 20), repmat('-', 1, 10), repmat('-', 1, 10), repmat('-', 1, 10));

for i = 1:numClasses
    currentClass = classNames{i};
    TP = sum(YValidation == currentClass & YPred == currentClass);
    FP = sum(YValidation ~= currentClass & YPred == currentClass);
    FN = sum(YValidation == currentClass & YPred ~= currentClass);
    
    
    precision(i) = TP / (TP + FP + eps); 
    recall(i) = TP / (TP + FN + eps);
    
    denominator = (precision(i) + recall(i) + eps);
    f1Score(i) = 2 * (precision(i) * recall(i)) / denominator;

    
    if isnan(precision(i))
        precision(i) = 0;
    end
    if isnan(recall(i))
        recall(i) = 0;
    end
    if isnan(f1Score(i))
        f1Score(i) = 0;
    end
    
    fprintf('%-20s %10.4f %10.4f %10.4f\n', currentClass, precision(i), recall(i), f1Score(i));
end

macroF1Score = mean(f1Score);
fprintf('\n----------------------------------------------------\n');
fprintf('Macro-Averaged F1-Score (Validation): %.4f\n', macroF1Score);
fprintf('----------------------------------------------------\n');

% Enregistrer le resultat

imdsTest = imageDatastore(testFolder, ...
    'IncludeSubfolders', false, ...
    'LabelSource', 'none');

% normaliser la taille d'entree
augimdsTest = augmentedImageDatastore(inputSize(1:2), imdsTest);


[YPredTest, ~] = classify(net, augimdsTest);

[~, testNames, ~] = cellfun(@fileparts, imdsTest.Files, 'UniformOutput', false);

jsonMap = containers.Map('KeyType', 'char', 'ValueType', 'char'); 
for i = 1:numel(testNames)
    key = testNames{i};
    value = char(YPredTest(i));
    jsonMap(key) = value;
end

jsonStr = jsonencode(jsonMap);

% enregistrer le fichier json (resultat)
outputFile = fullfile(pwd, "test_predictions_Restnet_RELU10-1.json"); 
fid = fopen(outputFile, 'w');
if fid == -1
    error('Erreur chemin,peut pas créer le fichier json');
end
fprintf(fid, '%s', jsonStr);
fclose(fid);
fprintf('\n json enregistré %s\n', outputFile);




