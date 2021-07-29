for i=1:10
cd('C:\Users\Olive\Desktop\#1_20db_H');
filename = ['20db_#1h_' num2str(i,'%2d') '.jpg'];
pred = imread(filename);
cd('C:\Users\Olive\Desktop\Fit2DGaussian Function to Data\Single_BB\20db_val_outputs');
filename = ['label_img_' num2str(i,'%2d') '.jpg'];
gt = imread(filename);

%gt=label_img_10;%eval(sprintf('label_img_%i', i));
%pred=x10db__5h_10;%eval(sprintf('x10db__5U_%i',i));
% figure()
% imshow(gt)
% figure()
% imshow(pred)
max_gt=max(gt,[],'all');
max_pre=max(pred,[],'all');

gt(gt>=(max_gt/2))=255;%real
gt(gt<(max_gt/2))=0;%TP
pred(pred>=(max_pre/2))=255;%pred
pred(pred<(max_pre/2))=0;
pre_own=pred;
gt_own=gt;

pre_own(pre_own==gt)=0;
TP1=pred-pre_own;
num_TP1=size(find(TP1));

gt_own(gt_own==pred)=0;%FP
FP=gt_own;
FN=pre_own;
% TP2=gt-gt_own;
% num_TP2=size(find(TP2));
num_FN=size(find(FN));
num_FP=size(find(FP));
a=[num_TP1(1) num_FN(1) num_FP(1)]
J=num_TP1(1)/(num_TP1(1)+num_FN(1)+num_FP(1))
end

cd('C:\Users\Olive\Desktop\Fit2DGaussian Function to Data')
%%
for i=1:10
% cd('C:\Users\Olive\Desktop\#1_20db_H');
% filename = ['20db_#1h_' num2str(i,'%2d') '.jpg'];
cd('C:\Users\Olive\Desktop\#1_20db_U');
filename = ['U_' num2str(i,'%2d') '.jpg'];
I = imread(filename);

BW=imbinarize(I);
s = regionprops(BW,'centroid');
centroids = cat(1,s.Centroid)
% figure
% imshow(BW)
% hold on
% plot(centroids(:,1),centroids(:,2),'b*')
% hold off
% title('ground true')
end