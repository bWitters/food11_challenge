%% -------------------------------------------------------------
%   TRAINING SCRIPT - FOOD11 WITH RESNET50 + EARLY STOPPING
%   + AFFICHAGE DISTRIBUTION AVANT / APRÈS SPLIT
%% -------------------------------------------------------------

clear all; close all; clc;

%% ----------------------- Dataset -----------------------------

trainFolder = "food-11/train";

imds = imageDatastore(trainFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', "foldernames");

%% ======= 📊 AFFICHAGE DISTRIBUTION AVANT SPLIT =======
figure;
histogram(imds.Labels)
title("Distribution des classes AVANT séparation")
xlabel("Classes"); ylabel("Nombre d'images");


%% ------------------ Train/Validation Split --------------------

[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.7, "randomized");

classNames = categories(imds.Labels);
numClasses = numel(classNames);

%% ======= 📊 AFFICHAGE DISTRIBUTION APRÈS SPLIT =======

figure;
subplot(1,2,1);
histogram(imdsTrain.Labels);
title("Distribution classes TRAIN");

subplot(1,2,2);
histogram(imdsValidation.Labels);
title("Distribution classes VALIDATION");


%% --------------------- Data Augmentation -----------------------

net = resnet50;
inputSize = net.Layers(1).InputSize;

augmenter = imageDataAugmenter( ...
    'RandXReflection', true, ...
    'RandRotation', [-10 10], ...
    'RandScale', [0.9 1.1], ...
    'RandXTranslation', [-5 5], ...
    'RandYTranslation', [-5 5], ...
    'RandXShear', [-5 5]);

augTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    DataAugmentation = augmenter);

augVal = augmentedImageDatastore(inputSize(1:2), imdsValidation);

%% ---------------------- Transfer Learning ----------------------

lgraph = layerGraph(net);
newFC = fullyConnectedLayer(numClasses, ...
    'Name', 'new_fc', ...
    'WeightLearnRateFactor', 10, ...
    'BiasLearnRateFactor', 10);

newClass = classificationLayer('Name', 'new_classoutput');

lgraph = replaceLayer(lgraph, 'fc1000', newFC);
lgraph = replaceLayer(lgraph, 'ClassificationLayer_fc1000', newClass);

%% ---------------------- Training Options -----------------------

miniBatchSize = 64;
initialLR = 1e-4;

options = trainingOptions("adam", ...
    "InitialLearnRate", initialLR, ...
    "L2Regularization", 0.0005, ...
    "MiniBatchSize", miniBatchSize, ...
    "MaxEpochs", 20, ...
    "Shuffle", "every-epoch", ...
    "ValidationData", augVal, ...
    "ValidationFrequency", 20, ...
    "ValidationPatience", 7, ...  
    "Verbose", false, ...
    "Plots", "training-progress");

%% ---------------------- GPU POWER LOGGER ------------------------
% (reste inchangé)

% Intervalle d'échantillonnage en millisecondes
sampleInterval = 500;

gpuPowerLog = fullfile(pwd, 'gpu_power_log.txt');
if isfile(gpuPowerLog), delete(gpuPowerLog); end
[status, ~] = system('nvidia-smi -L');
if status == 0
    fprintf("✔ GPU détecté, lancement du monitoring de puissance...\n");
    system(sprintf('start /B nvidia-smi --loop-ms=%d --query-gpu=power.draw --format=csv,noheader,nounits > "%s"', ...
        sampleInterval, gpuPowerLog));
else
    warning("⚠ Impossible d'accéder à nvidia-smi. Monitoring désactivé.");
    gpuPowerLog = "";
end

trainTimer = tic;

%% --------------------------- Train -----------------------------

netTrained = trainNetwork(augTrain, lgraph, options);

%% ---------------------- FIN TRAINING --------------------------

trainTime = toc(trainTimer);
fprintf("⏱ Temps total d'entraînement : %.2f s (%.2f min)\n", trainTime, trainTime/60);
system('taskkill /F /IM nvidia-smi.exe > nul 2>&1');


%% --------------------- Save with timestamp ----------------------

timestamp = datetime("now");
timestamp.Format = 'yyyy-MM-dd_HH-mm-ss';
filename = "trainedNet_" + string(timestamp) + ".mat";

save(filename, "netTrained", "classNames");

fprintf("\n✔ Réseau sauvegardé dans : %s\n", filename);
