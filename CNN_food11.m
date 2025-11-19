inputSize = [28 28 3];  
numClasses = numel(categories(imdsTrain.Labels));


% appliquer augmentation avancée uniquement sur le train 
imdsTrain.ReadFcn = @(filename) advancedAugmenter(filename, inputSize(1:2));
% validation normalisée
imdsValidation.ReadFcn = @(filename) imresize(imread(filename), inputSize(1:2));

augimdsTrain = augmentedImageDatastore(inputSize, imdsTrain);
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
    MiniBatchSize=64, ...
    InitialLearnRate=1e-4, ...
    ValidationData=imdsValidation, ...
    ValidationFrequency=50, ...
    Plots="training-progress", ...
    Verbose=false);

% --- Entraînement ---
net = trainNetwork(augimdsTrain, layers, options);