clear all;clc;close all;


%%% 62     5     3    60     7
%%% First load the power spectrum .mat file, the alpha rythem is the most
%%% representative one
load('Rhythm.mat');

Rhythm=squeeze(mean(mean(Rhythm(:,:,:,:,[3 4]),5),4));%62 5 3
%%%%  alpha1 (8~10.5 Hz), alpha2 (10.5~13 Hz),

%%%% Plot the spectral topography
load chanlocs62.mat;
type={'EO','EC','Music','Memory','Math'};
figure(3)
for i=1:3
    for j=1:5
        subplot(3,5,i*5+j-5)
        topoplot(squeeze(Rhythm(:,j,i)),chanlocs,'maplimits',[-6 18]);
        if i==1
            title(type{j});
        end
    end
end