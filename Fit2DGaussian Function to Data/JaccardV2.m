% JaccardV2. depends on real number of BB, not pixels
% 判断质心坐标的距离，Th内距离的坐标认定为正确： TP
record_J=[];
roc_TP=[];
roc_FP=[];
roc_FN=[];rrecall=[];
pprecision=[];
mmissrate=[];
%figure()

for th=1%[0:0.01:1 1:100:1000]
for i=1:30
cd('C:\Users\Olive\Desktop\H10db_3px');
filename = ['H10db_' num2str(i,'%2d') '.jpg'];
pred = imread(filename);
cd('C:\Users\Olive\Desktop\10_10db_out_1px');
filename = ['label_img_' num2str(i,'%2d') '.jpg'];
gt = imread(filename);

% figure()
% imshow(gt)
% figure()
% imshow(pred)
BW_pred=imbinarize(pred);
% figure()
% imshow(BW_pred)
stats_pred = regionprops('table',BW_pred,'Centroid',...
    'MajorAxisLength','MinorAxisLength');
centers_pred = stats_pred.Centroid;
diameters_pred= mean([stats_pred.MajorAxisLength stats_pred.MinorAxisLength],2);
radii_pred = diameters_pred/2;

BW_gt=imbinarize(gt);
% figure()
% imshow(BW_gt)
stats_gt = regionprops('table',BW_gt,'Centroid',...
    'MajorAxisLength','MinorAxisLength');
centers_gt = stats_gt.Centroid;
diameters_gt= mean([stats_gt.MajorAxisLength stats_gt.MinorAxisLength],2);
radii_gt = diameters_gt/2;

% Th=82.8 um
%Th_max= Th/fact_1024=48.1395
% count the correct coordinates; the minmum pixel distance is 48.1395
real_sz_pred=size(centers_pred,1);
real_sz_gt=size(centers_gt,1);
%th=0;
for m=1:size(centers_pred,1)
    j=size(centers_gt,1);
    dist(m)=norm(centers_pred(m,:)-centers_gt(j,:));
    while (dist(m)>th) && (j>=1)
        dist(m)=norm(centers_pred(m,:)-centers_gt(j,:));
           j=j-1;   
    end  
    if dist(m)<=th
       j=j+1;
       if j >size(centers_gt,1)
           j=size(centers_gt,1);
       end
       centers_gt(j,:)=[]; %centers_gt 目前剩下多少都不管了，用来安全计算每一个center_pred
       
    end
end

FP=size(dist(dist>th),2);%detect more point

TP=size(dist(dist<=th),2);%detect real point
FN=size(centers_gt,1);%missed point

% fprintf('index %d\n',i)
% fprintf('FN1==FN2:	%f\n ',FN1==FN2);
% fprintf('FN1:	%f\n ',FN1);
% fprintf('FN2:	%f \n',FN2);fprintf(' \n');
Jaccard=TP/(TP+FP+FN);
record_J=[record_J;Jaccard];
roc_TP=[roc_TP,TP];
roc_FN=[roc_FN,FN];
roc_FP=[roc_FP,FP];

end
recall=sum(roc_TP)/(sum(roc_TP)+sum(roc_FN));
precision=sum(roc_TP)/(sum(roc_TP)+sum(roc_FP));
missrate=sum(roc_FN)/(sum(roc_TP)+sum(roc_FN));
rrecall=[rrecall; recall];
pprecision=[pprecision;precision];
mmissrate=[mmissrate;missrate];

end
% xre=0:0.1:1;
% yre=1-xre;
% 
 %plot(mmissrate,pprecision,'*r');
% hold on
% plot(xre,yre,'-g');
% hold off
ylabel('pprecision','FontSize',16)
xlabel('th','FontSize',16)
axis([0 1 0 1])
% xlim=([0 1]);
% ylim=([0 1]);

cd('C:\Fit2DGaussian Function to Data')