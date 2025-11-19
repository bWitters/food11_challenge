inputSize = [224 224 3];
numClasses = numel(categories(imdsTrain.Labels));

% appliquer augmentation avancée uniquement sur le train 
imdsTrain.ReadFcn = @(filename) advancedAugmenter(filename, inputSize(1:2));
% validation normalisée
imdsValidation.ReadFcn = @(filename) imresize(imread(filename), inputSize(1:2));

augimdsTrain = augmentedImageDatastore(inputSize, imdsTrain);
augimdsValidation = augmentedImageDatastore(inputSize, imdsValidation);

% chargement du réseau enet
net = efficientnetb0;
lgraph = layerGraph(net);

newLayers = [
    fullyConnectedLayer(numClasses,"Name","fc_food11")
    softmaxLayer("Name","softmax_food11")
    classificationLayer("Name","classoutput_food11")
];

lgraph = replaceLayer(lgraph, "efficientnet-b0|model|head|dense|MatMul", newLayers(1));
lgraph = replaceLayer(lgraph, "Softmax", newLayers(2));
lgraph = replaceLayer(lgraph, "classification", newLayers(3));

% Geler les couches
layers = lgraph.Layers;
connections = lgraph.Connections;

for i = 1:numel(layers)-3
    if isprop(layers(i), "WeightLearnRateFactor")
        layers(i).WeightLearnRateFactor = 0;
        layers(i).BiasLearnRateFactor = 0;
    end
end

% Options d'entraînement
options = trainingOptions("adam", ...
    MiniBatchSize=64, ...
    MaxEpochs=12, ...
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
