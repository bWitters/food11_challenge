%% -------------------------------------------------------------
%   TEST SCRIPT - SIMPLE JSON DICTIONARY FORMAT
%   Output JSON : { "0": "bread", "1": "fried", ... }
%% -------------------------------------------------------------

clear all; close all; clc;

%% ---------------------- Charger le réseau ----------------------
modelFile = "trainedNet_2025-11-19_18-32-41";   % <-- remplace par ton modèle
modelFileExt = modelFile + ".mat";
load(modelFileExt, "netTrained", "classNames");

%% --------------------------- Jeu de test ------------------------

inputSize = netTrained.Layers(1).InputSize;

testFolder = "food-11/test";    % dossier test SANS labels

if ~isfolder(testFolder)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', myFolder);
  uiwait(warndlg(errorMessage));
  return;
end
filePattern = fullfile(testFolder, '*.jpg');
jpegFiles = dir(filePattern);

results = containers.Map;

%% -------------------------- Prédictions -------------------------

for k = 1:length(jpegFiles)
  baseFileName = jpegFiles(k).name;
  fullFileName = fullfile(testFolder, baseFileName);
  fprintf(1, 'Now reading %s\n', fullFileName);
  imageArray = imread(fullFileName);
  YPred = scores2label(predict(netTrained, im2double(imresize(imageArray,inputSize(1:2)))),classNames);
  picName = split(baseFileName,".");
  results(picName{1}) = YPred;
end

%% --------------------------- Export JSON -----------------------------

jsonText = jsonencode(results);

timestamp = datetime("now");
timestamp.Format = 'yyyy-MM-dd_HH-mm-ss';
filename = modelFile + string(timestamp) + ".json";

fid = fopen(filename, 'w');
fprintf(fid, "%s", jsonText);
fclose(fid);

fprintf("✔ JSON généré : %s\n", filename);
