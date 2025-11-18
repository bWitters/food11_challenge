sampleInterval = 500; 
gpuPowerLog = fullfile(pwd,'gpu_power_log.txt');
system(sprintf('start /B nvidia-smi --loop-ms=%d --query-gpu=power.draw --format=csv,noheader,nounits > "%s"', sampleInterval, gpuPowerLog));

if isfile(gpuPowerLog)
powerData = readmatrix(gpuPowerLog);
avgPower = mean(powerData);
energyWh = avgPower * (trainTime/3600); % Wh
fprintf('Average GPU Power: %.2f W\n', avgPower);
fprintf('Estimated GPU Energy Consumption: %.4f Wh\n', energyWh);
else
warning('GPU power log not found. Energy not computed.');
end
