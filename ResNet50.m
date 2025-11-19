%% -------------------------------------------------------------
%   TRAINING SCRIPT - FOOD11 WITH RESNET50 + EARLY STOPPING
%   Sauvegarde automatique du réseau
%   Auteurs : Benjamin WITTERS, Gia HOANG, Yafei XU, Jingwen SHEN, ChatGPT
%% -------------------------------------------------------------

clear all; close all; clc;

%% ----------------------- Dataset -----------------------------

trainFolder = "food-11/train";

imds = imageDatastore(trainFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', "foldernames");

[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.7, "randomized");

classNames = categories(imds.Labels);
numClasses = numel(classNames);

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

% Early stopping requires MATLAB R2023b+
options = trainingOptions("adam", ...
    "InitialLearnRate", initialLR, ...
    "L2Regularization", 0.0005, ...
    "MiniBatchSize", miniBatchSize, ...
    "MaxEpochs", 20, ...
    "Shuffle", "every-epoch", ...
    "ValidationData", augVal, ...
    "ValidationFrequency", 20, ...
    "ValidationPatience", 7, ...       % EARLY STOPPING
    "Verbose", false, ...
    "Plots", "training-progress");

%% ---------------------- GPU POWER LOGGER ------------------------

% Intervalle d'échantillonnage en millisecondes
sampleInterval = 500;

% Fichier où sera enregistré la puissance GPU
gpuPowerLog = fullfile(pwd, 'gpu_power_log.txt');

% Supprime un ancien fichier si il existe
if isfile(gpuPowerLog)
    delete(gpuPowerLog);
end

% Test si nvidia-smi est disponible
[status, ~] = system('nvidia-smi -L');
if status == 0
    fprintf("✔ GPU détecté, lancement du monitoring de puissance...\n");

    % Commande Windows pour exécuter nvidia-smi en tâche de fond
    % (pour Linux : adapter avec nohup &)
    system(sprintf('start /B nvidia-smi --loop-ms=%d --query-gpu=power.draw --format=csv,noheader,nounits > "%s"', ...
        sampleInterval, gpuPowerLog));
else
    warning("⚠ Impossible d'accéder à nvidia-smi. Monitoring de puissance désactivé.");
    gpuPowerLog = "";
end

% Démarrer un chronomètre (pour calculer énergie)
trainTimer = tic;

%% --------------------------- Train -----------------------------

netTrained = trainNetwork(augTrain, lgraph, options);

%% ---------------------- FIN TRAINING → STOP LOGGING GPU -----------------------

trainTime = toc(trainTimer);  % temps total en secondes

fprintf("⏱ Temps total d'entraînement : %.2f s (%.2f min)\n", trainTime, trainTime/60);

% Arrêter le process nvidia-smi
% (on tue le processus nvidia-smi en arrière-plan)
system('taskkill /F /IM nvidia-smi.exe > nul 2>&1');

pause(0.2); % attendre une fraction de seconde que le fichier soit écrit

if gpuPowerLog ~= "" && isfile(gpuPowerLog)
    try
        powerData = readmatrix(gpuPowerLog);

        % Filtrer puissances invalides
        powerData = powerData(~isnan(powerData) & powerData > 0);

        if ~isempty(powerData)
            avgPower = mean(powerData);           % W
            energyWh = avgPower * (trainTime/3600); % Wh

            fprintf("\n=== GPU POWER REPORT ===\n");
            fprintf("Average GPU Power : %.2f W\n", avgPower);
            fprintf("Estimated GPU Energy Consumption : %.4f Wh\n", energyWh);
            fprintf("==========================\n\n");
        else
            warning("⚠ Aucun échantillon valide dans le log GPU.");
        end

    catch ME
        warning("Erreur lors de la lecture du log GPU : %s", ME.message);
    end
else
    warning("GPU power log not found. Energy not computed.");
end

%% --------------------- Save with timestamp ----------------------

timestamp = datetime("now");
timestamp.Format = 'yyyy-MM-dd_HH-mm-ss';
filename = "trainedNet_" + string(timestamp) + ".mat";

save(filename, "netTrained", "classNames");

fprintf("\n✔ Réseau sauvegardé dans : %s\n", filename);