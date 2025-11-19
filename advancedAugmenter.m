function I = advancedAugmenter(filename, targetSize)

    I = imread(filename);
    I = im2single(I);
    I = imresize(I, targetSize);

    % Rotation
    angle = randi([-25 25]);
    I = imrotate(I, angle, 'bilinear', 'crop');

    % Flip
    if rand > 0.5, I = fliplr(I); end
    if rand > 0.5, I = flipud(I); end

    % Translation
    dx = randi([-15 15]);
    dy = randi([-15 15]);
    I = imtranslate(I, [dx dy]);

    % Couleur
    hsv = rgb2hsv(I);
    hsv(:,:,1) = hsv(:,:,1) + (rand-0.5)*0.1;
    hsv(:,:,2) = hsv(:,:,2) .* (0.7 + 0.6*rand);
    I = hsv2rgb(hsv);

    % Contraste
    I = imadjust(I, stretchlim(I, 0.01 + rand*0.1), []);

    % Bruit
    if rand > 0.5
        I = imnoise(I, "gaussian", 0, 0.002 + 0.008*rand);
    end

    % Gamma
    gamma = 0.7 + 0.6*rand;
    I = imadjust(I, [], [], gamma);

    % Flou
    if rand > 0.7
        I = imgaussfilt(I, 0.5 + rand);
    end

    % Cutout
    if rand > 0.5
        sz = size(I);
        cut = randi([20 50]);
        x = randi([1 sz(2)-cut]);
        y = randi([1 sz(1)-cut]);
        I(y:y+cut, x:x+cut, :) = rand;
    end
end
