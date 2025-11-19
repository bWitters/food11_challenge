%% -------------------------------------------------------------------
%   BEST F1 TRAINING SCRIPT - FOOD11
%   EfficientNetB0 (imagePretrainedNetwork) + heavy augmentation
%   Early stopping + CLR + timestamped save
%% -------------------------------------------------------------------

clear all; close all; clc;

%% ---------------------------- Dataset ------------------------------

trainFolder = "food-11/train";

imds = imageDatastore(trainFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', "foldernames");

[imdsTrain, imdsVal] = splitEachLabel(imds, 0.75, "randomized");

classNames = categories(imds.Labels);
numClasses = numel(classNames);

%% -------------------- Load EfficientNetB0 --------------------------

net = imagePretrainedNetwork(NumClasses=numClasses);  
inputSize = net.Layers(1).InputSize;

%% -------------------------- Augmentation ---------------------------

augmenter = imageDataAugmenter( ...
    'RandXReflection', true, ...
    'RandRotation', [-15 15], ...
    'RandXTranslation', [-15 15], ...
    'RandYTranslation', [-15 15], ...
    'RandScale', [0.85 1.15], ...
    'RandXShear', [-10 10], ...
    'RandYShear', [-10 10]);

augTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, DataAugmentation = augmenter);
augVal   = augmentedImageDatastore(inputSize(1:2), imdsVal);

%% ---------------- Replace Classification Head ----------------------

net = setLearnRateFactor(net,"conv10/Weights",10);
net = setLearnRateFactor(net,"conv10/Bias",10);

%% ------------------------- Training Options ------------------------

miniBatchSize = 32;

options = trainingOptions("sgdm", ...
    InitialLearnRate = 3e-4, ...
    Momentum = 0.9, ...
    L2Regularization = 0.0005, ...
    MiniBatchSize = miniBatchSize, ...
    MaxEpochs = 40, ...
    Shuffle = "every-epoch", ...
    ValidationData = augVal, ...
    ValidationFrequency = 50, ...
    ValidationPatience = 5, ...   % early stopping
    Plots = "training-progress", ...
    Verbose = false);

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

%% ------------------------------ Train ------------------------------

netBestF1 = trainnet(augTrain,net,"crossentropy",options);

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

%% --------------------------- Save Model ----------------------------

timestamp = datetime("now");
timestamp.Format = 'yyyy-MM-dd_HH-mm-ss';
filename = "bestF1_Food11_pretrained_" + string(timestamp) + ".mat";

save(filename, "netBestF1", "classNames");

fprintf("\n✔ Modèle sauvegardé : %s\n", filename);
