channels = [];
for c = DeepMIMO_dataset{1}.user
    x = c{1}.channel; % only use the results of the first subcarrier and the first OFDM slot
    if size(x,1) ~= dataset_params.CDL_5G.num_slots*14
        disp(size(x,1));
        continue;
    end
    channels = cat(1,channels,x(1:14:end,:,:,1));
    disp(size(channels));
    
end

fileName = strcat('DeepMIMO_dataset_new/',dataset_params.scenario,'_path',num2str(dataset_params.num_paths),'_seed',num2str(dataset_params.seed),'.mat');
save(fileName,"channels","dataset_params");