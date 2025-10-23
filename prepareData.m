clear all
close all

folderName = "food-11\train";

imds = imageDatastore(folderName, ...
    IncludeSubfolders=true, ...
    LabelSource="foldernames");

outputSize = [32 32 3];

classNames = categories(imds.Labels);
numClasses = numel(classNames);

[imdsTrain,imdsValidation] = splitEachLabel(imds,0.9,0.1,"randomized");

auimdsTrain = augmentedImageDatastore(outputSize,imdsTrain);
auimdsValidation = augmentedImageDatastore(outputSize,imdsValidation);