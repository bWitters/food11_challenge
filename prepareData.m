clear all
close all

folderName = "food-11\train"

imds = imageDatastore(folderName, ...
    IncludeSubfolders=true, ...
    LabelSource="foldernames");

% numImages = numel(imds.Labels);
% idx = randperm(numImages,16);
% I = imtile(imds,Frames=idx);
% figure
% imshow(I)
% 
% classNames = categories(imds.Labels);
% numClasses = numel(classNames);
% 
% [imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,0.3,"randomized");

% Afficher le nombre d’images par classe
tbl = countEachLabel(imds);
disp(tbl);

% Diviser les données : 80 % entraînement, 20 % validation
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.8, 'randomized');

% Vérifier les tailles
disp('Nombre d’images dans l’ensemble d’entraînement :');
disp(countEachLabel(imdsTrain));
disp('Nombre d’images dans l’ensemble de validation :');
disp(countEachLabel(imdsValidation));

% Afficher quelques exemples de chaque ensemble
figure;
subplot(1,2,1);
imshow(readimage(imdsTrain, 1));
title("Exemple - Entraînement (" + string(imdsTrain.Labels(1)) + ")");

subplot(1,2,2);
imshow(readimage(imdsValidation, 1));
title("Exemple - Validation (" + string(imdsValidation.Labels(1)) + ")");
