clear all
close all

folderName = "food-11\train"

imds = imageDatastore(folderName, ...
    IncludeSubfolders=true, ...
    LabelSource="foldernames");

numImages = numel(imds.Labels);
idx = randperm(numImages,16);
I = imtile(imds,Frames=idx);
figure
imshow(I)

classNames = categories(imds.Labels);
numClasses = numel(classNames);

[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,0.3,"randomized");