clear all;clc;close all;


% %%% step one£ºcalculate the power spectrum
% %%%% Px 4097 62 5 3 60 [frequency electrodes sates sessions number of participants]

load('PxSensor.mat')

Pxm=Px(:,[4 13 23],:,:,:);

Pxxmm=squeeze(mean(Pxm,5));%4097 3 5 3
Pxxms=squeeze(std(Pxm,0,5))./sqrt(60);%4097 3 5 3

xtime=[0:4096]./4096*250;% Set the corresponding timeline here
xtime12=[xtime xtime(end:-1:1)];
colorM=[0 1 1;1 1 0;1 0 0;0 1 0;1 0 1];% Set the color of the mean line here
colorS=colorM*0.5+0.5;% Set the color of the standard error box here
figure(1)
cha={'Fz','Cz','Pz'}; 
for ii=1:3
    for j=1:3
        subplot(3,3,ii*3+j-3);
        ERP=squeeze(Pxxmm(:,ii,:,j));
        ERPse=squeeze(Pxxms(:,ii,:,j));
        ERP1=ERP+ERPse;
        ERP2=ERP-ERPse;
        ERP12=[ERP1;ERP2(size(ERP2,1):-1:1,:)];
        for i=1:5
            plot(xtime,ERP(:,i),'color',colorM(i,:),'LineWidth',2);% set the mean line
            hold on;
        end
        grid on;
        if j==1 & ii==1
            legend({'EO','EC','Music','Memory','Math'})
        end
        xlim([2 47]);
        ylim([-30 20]);
        title([cha{ii},' - session',num2str(j)]);
    end
end

