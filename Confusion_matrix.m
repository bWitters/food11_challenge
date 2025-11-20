%% ---------------------------------------------------------------
%   EVALUATION SCRIPT FOR FOOD-11 (PER-CLASS METRICS)
%   Computes: Confusion Matrix, Precision, Recall, F1 per class
%% ---------------------------------------------------------------

clear all; close all; clc;

%% ---------------------- Load Trained Model ----------------------

modelFile = "trainedNet_2025-11-19_18-32-41";   % <-- remplace par ton modèle
modelFileExt = modelFile + ".mat";
load(modelFileExt, "netTrained", "classNames");

fprintf("✔ Loaded model: %s\n\n", modelFile);

%% ------------------------- Load Validation Set -------------------

trainFolder = "food-11/train";

imds = imageDatastore(trainFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', "foldernames");

[~, imdsVal] = splitEachLabel(imds, 0.8, "randomized");

numClasses = numel(classNames);

fprintf("Validation images: %d\n", numel(imdsVal.Labels));

%% ---------------------- Prepare Data for Network -----------------

inputSize = netTrained.Layers(1).InputSize;

augVal = augmentedImageDatastore(inputSize(1:2), imdsVal);

%% -------------------------- Run Classification --------------------

fprintf("\nClassifying validation set...\n");

[YPred, ~] = classify(netTrained, augVal);
YVal = imdsVal.Labels;

accuracy = mean(YPred == YVal);
fprintf('Validation Accuracy: %.2f%%\n', accuracy*100);

%% -------------------------- Confusion Matrix ----------------------

figure;
plotconfusion(YVal, YPred);
title('Confusion Matrix – Validation');

%% --------------------------- Metrics ------------------------------

precision = zeros(numClasses,1);
recall    = zeros(numClasses,1);
f1        = zeros(numClasses,1);

for i = 1:numClasses
    class = classNames{i};

    TP = sum(YTrue == class & YPred == class);
    FP = sum(YTrue ~= class & YPred == class);
    FN = sum(YTrue == class & YPred ~= class);

    precision(i) = TP / (TP + FP + eps);
    recall(i)    = TP / (TP + FN + eps);
    f1(i)        = 2 * precision(i) * recall(i) / (precision(i) + recall(i) + eps);
end

%% ------------------------ Display Results -------------------------

T = table(classNames, precision, recall, f1, ...
    'VariableNames', {'Class', 'Precision', 'Recall', 'F1_Score'});

disp("--------------------------------------------------------------");
disp("                 PER-CLASS VALIDATION METRICS");
disp("--------------------------------------------------------------");
disp(T);

macroF1 = mean(f1);

fprintf("\nMacro-F1 Score: %.4f\n", macroF1);

%% ------------------- Highlight Weak Classes -----------------------

fprintf("\n📉 **Weakly Performing Classes (<0.70 F1)**:\n");

weakIdx = find(f1 < 0.70);

if isempty(weakIdx)
    fprintf("✔ None — all classes above 0.70 F1\n");
else
    for i = weakIdx'
        fprintf(" - %s : F1 = %.3f\n", classNames{i}, f1(i));
    end
end
