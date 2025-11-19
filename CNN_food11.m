inputSize = [28 28 3];  
numClasses = numel(categories(imdsTrain.Labels));

% --- Redimensionnement automatique des images ---
imdsTrain.ReadFcn = @(filename) imresize(imread(filename), inputSize(1:2));
imdsValidation.ReadFcn = @(filename) imresize(imread(filename), inputSize(1:2));

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

% --- Réseau CNN ---
layers = [
    imageInputLayer(inputSize)

    convolution2dLayer(5, 20)
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 40)
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2, 'Stride', 2)

    fullyConnectedLayer(100)
    reluLayer

    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer
];

% --- Options d'entraînement --- 
options = trainingOptions("sgdm", ...
    MaxEpochs=16, ...
    MiniBatchSize=100, ...
    L2Regularization=0.00001, ...
    InitialLearnRate=1e-4, ...
    ValidationData=imdsValidation, ...
    ValidationFrequency=50, ...
    Plots="training-progress", ...
    Verbose=false);

% --- Entraînement ---
net = trainNetwork(imdsTrain, layers, options);
