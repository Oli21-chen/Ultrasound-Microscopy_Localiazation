cd('C:\Users\Olive\Desktop\Fit2DGaussian Function to Data\16_bnum\16_pred\HRNet');
Files=dir('frame*.*');
for k=1%:length(Files)
 FileNames = sprintf('frame_%d.jpg', k-1);
 %FileNames=Files(k).name;
 I= imread(FileNames);
 BW=imbinarize(I,0.4);
 s = regionprops(BW,'centroid');
 centroids = cat(1,s.Centroid);
 writematrix(centroids','16_HRNPred.xlsm','WriteMode','append');
end


%% %%%
cd('C:\Users\Olive\Desktop\Fit2DGaussian Function to Data\20_bnum\outputs')
Files=dir('label*.*');
for k=1%:length(Files)
FileNames = sprintf('label_img_%d.jpg', k);
 I= imread(FileNames);
end
BW=imbinarize(I,0.5);
s = regionprops(BW,'centroid');
centroids = cat(1,s.Centroid);
figure
imshow(BW)
hold on
plot(centroids(:,1),centroids(:,2),'b*')
hold off
title('ground true')

cd('C:\Users\Olive\Desktop\Fit2DGaussian Function to Data\20_bnum\20_pred\HRNet')
Files=dir('frame*.*');
for k=1%:length(Files)
 FileNames = sprintf('frame_%d.jpg', k-1);
 I= imread(FileNames);
 figure()
 imshow(I)
end
BW=imbinarize(I,0.5);
s = regionprops(BW,'centroid');
pred_centroids = cat(1,s.Centroid);
figure
imshow(BW)
hold on
plot(pred_centroids(:,1),pred_centroids(:,2),'b*')
hold off
title('Pred')

