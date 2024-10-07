clear all; clc;
% ----------------- Add the path of DeepMIMO function --------------------%
addpath('DeepMIMO_functions')

% -------------------- DeepMIMO Dataset Generation -----------------------%
scenariolist= {'I2_28B'};
for sidx=1:length(scenariolist)
    for seed = [1111,1212]
        rng(seed,"twister");
        % Load Dataset Parameters
        disp(scenariolist{sidx});
        dataset_params = ConfigDeepMIMOParams(scenariolist{sidx});
        dataset_params.seed = seed;
        [DeepMIMO_dataset, dataset_params] = DeepMIMO_generator(dataset_params);
        save_dataset;
    end
end

% -------------------------- Output Examples -----------------------------%
% DeepMIMO_dataset{i}.user{j}.channel % Channel between BS i - User j
% %  (# of User antennas) x (# of BS antennas) x (# of OFDM subcarriers)
%
% DeepMIMO_dataset{i}.user{j}.params % Parameters of the channel (paths)
% DeepMIMO_dataset{i}.user{j}.LoS_status % Indicator of LoS path existence
% %     | 1: LoS exists | 0: NLoS only | -1: No paths (Blockage)|
%
% DeepMIMO_dataset{i}.user{j}.loc % Location of User j
% DeepMIMO_dataset{i}.loc % Location of BS i
%
% % BS-BS channels are generated only if (params.enable_BSchannels == 1):
% DeepMIMO_dataset{i}.basestation{j}.channel % Channel between BS i - BS j
% DeepMIMO_dataset{i}.basestation{j}.loc
% DeepMIMO_dataset{i}.basestation{j}.LoS_status
%
% -------------------------- Dynamic Scenario ----------------------------%
%
% DeepMIMO_dataset{f}{i}.user{j}.channel % Scene f - BS i - User j
% % Every other command applies as before with the addition of scene ID
% params{f} % Parameters of Scene f
%
% ------------------------------------------------------------------------%