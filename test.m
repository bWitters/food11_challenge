%  Réseau ResNet18 sans poids pré-entraînés

inputSize = [224 224 3];
numClasses = numel(categories(imdsTrain.Labels));

% Redimensionnement
imdsTrain.ReadFcn = @(filename) imresize(imread(filename), inputSize(1:2));
imdsValidation.ReadFcn = @(filename) imresize(imread(filename), inputSize(1:2));

% Charger ResNet18 sans poids
netStruct = resnet18('Weights','none');
disp("Type de l’objet renvoyé par resnet18:");
disp(class(netStruct));

lgraph = netStruct;

% Ajout d'images
imageAugmenter = imageDataAugmenter( ...
    'RandRotation', [-20 20], ...
    'RandXReflection', true, ...
    'RandYReflection', true, ...
    'RandXTranslation', [-10 10], ...
    'RandYTranslation', [-10 10], ...
    'RandScale', [0.8 1.2]);

augimdsTrain = augmentedImageDatastore(inputSize, imdsTrain, 'DataAugmentation', imageAugmenter);
augimdsValidation = augmentedImageDatastore(inputSize, imdsValidation);

% Remplacer les dernières couches
newLayers = [
    fullyConnectedLayer(numClasses, "Name", "fc_food11")
    softmaxLayer("Name", "softmax_food11")
    classificationLayer("Name", "classoutput_food11")
];

lgraph = replaceLayer(lgraph, "fc1000", newLayers(1));
lgraph = replaceLayer(lgraph, "prob", newLayers(2));
lgraph = replaceLayer(lgraph, "ClassificationLayer_predictions", newLayers(3));

% Options d'entraînement
options = trainingOptions("sgdm", ...
    MiniBatchSize=500, ...
    MaxEpochs=16, ...
    InitialLearnRate=1e-4, ...
    L2Regularization=0.0005, ...  % <-- régularisation
    Shuffle="every-epoch", ...
    ValidationData=augimdsValidation, ...
    ValidationFrequency=50, ...
    Plots="training-progress", ...
    Verbose=false);

% Entraînement
trainedNet = trainNetwork(augimdsTrain, lgraph, options);
% trainedNet = trainNetwork(imdsTrain, lgraph, options);

% Évaluation
YPred = classify(trainedNet, imdsValidation);
YTrue = imdsValidation.Labels;
accuracy = mean(YPred == YTrue);
fprintf('\n Précision sur validation : %.2f %%\n', accuracy * 100);
