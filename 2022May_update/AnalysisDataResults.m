%% for Jaccard
SNR=[10, 20, 30];
uJa_1=[1, 1, 1];
hJa_1=[1, 1, 1];
uJa_5=[0.9833, 0.951, 0.9586];
hJa_5=[0.9667, 0.9644, 0.9468];
uJa_10=[0.9159, 0.9387, 0.9356];
hJa_10=[0.9483, 0.9493, 0.9576];
figure()
plot(SNR,uJa_1,'--o',SNR,uJa_5,':x',SNR,uJa_10,'-+','MarkerSize',10,'LineWidth',3);
ylim([0.9 ,1.05]);
xlabel('SNR')
ylabel('Jaccard Index')
title('Jaccard results')
hold on
plot(SNR,hJa_1,'--o',SNR,hJa_5,':x',SNR,hJa_10,'-+','MarkerSize',10,'LineWidth',3);
legend('0.32 MBs/(mm*mm) for U','1.6 MBs/(mm*mm) for U','3.2 MBs/(mm*mm)for U',...
    '0.32 MBs/(mm*mm) for A','1.6 MBs/(mm*mm) for A','3.2 MBs/(mm*mm) for A');
hold off
set(gca,'FontSize',15)

%% for RMSE
SNR=[10, 20, 30];
uRM_1=[2.446, 0.794, 1.4295];
hRM_1=[1.7932,0.9811,1.7439];
uRM_5=[1.4702, 0.6235, 1.3094];
hRM_5=[1.0678, 0.8530,1.1453];
uRM_10=[1.9901, 1.81,2.2808];
hRM_10=[1.7318, 1.6167, 3.4539];

figure()
plot(SNR,uRM_1,'--o',SNR,uRM_5,':x',SNR,uRM_10,'-+','MarkerSize',10,'LineWidth',3);
%ylim([0.9 ,1.05]);
xlabel('SNR')
ylabel('RMSE')
title('RMSE results')
hold on
plot(SNR,hRM_1,'--o',SNR,hRM_5,':x',SNR,hRM_10,'-+','MarkerSize',10,'LineWidth',3);
legend('0.32 MBs/(mm*mm) for U','1.6 MBs/(mm*mm) for U','3.2 MBs/(mm*mm)for U',...
    '0.32 MBs/(mm*mm) for A','1.6 MBs/(mm*mm) for A','3.2 MBs/(mm*mm) for A');
hold off
set(gca,'FontSize',15)

%% Multibubble precision_ Auto-HRNet
x= [0.32,1.6,3.2];
la_10db=[2.415, 45.712, 64.142];
la_20db=[13.219,28.1968,41.443];
la_30db=[7.224, 54.477,54.084];
yyaxis left
plot (x,la_10db)
title('Plots with Different y-Scales')
xlabel('Values from 0 to 25')
ylabel('Left Side')


yyaxis right
ylim([10 20])
ylabel('Right Side')