inputSize = [32 32 3];
numClasses = 11;

layers = [
    imageInputLayer([32 32 3], "Name", "input")

    % Bloc convolution + normalisation + ReLU
    convolution2dLayer(3, 32, "Padding", "same", "Name", "conv_1")
    batchNormalizationLayer("Name", "bn_1")
    reluLayer("Name", "relu_1")

    maxPooling2dLayer(2, "Stride", 2, "Name", "pool_1")

    convolution2dLayer(3, 64, "Padding", "same", "Name", "conv_2")
    batchNormalizationLayer("Name", "bn_2")
    reluLayer("Name", "relu_2")

    maxPooling2dLayer(2, "Stride", 2, "Name", "pool_2")

    convolution2dLayer(3, 128, "Padding", "same", "Name", "conv_3")
    batchNormalizationLayer("Name", "bn_3")
    reluLayer("Name", "relu_3")

    fullyConnectedLayer(256, "Name", "fc_1")
    reluLayer("Name", "relu_4")

    fullyConnectedLayer(numClasses, "Name", "fc_2")
    softmaxLayer("Name", "softmax")
];

analyzeNetwork(layers)

% Set training options and train the network
miniBatchSize = 500;

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',20, ...
    'Shuffle','every-epoch', ...
     'MiniBatchSize', miniBatchSize,...
     'ValidationData',auimdsValidation,...
     'ValidationFrequency',40,...
    'Verbose',false, ...
    'Plots','training-progress', ...
    'Metrics','accuracy');

net = trainnet(auimdsTrain,layers,"crossentropy",options);