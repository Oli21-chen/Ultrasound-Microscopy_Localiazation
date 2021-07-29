%% Multibubble generator%%
% bubble num is 15, 16, 17, 18, 19, 20
% pic num of each cond: 50

%%

real_sz=1.7664e+3;%um
fact_128=real_sz/128;
fact_1024=real_sz/1024;
for j=1:10
rx=[];
ry=[];
[X,Y] = meshgrid(-63:1:64);
xdata=zeros(size(X,1),size(X,2),2);
xdata(:, :, 1) = X;
xdata(:, :, 2) = Y; 
[X_gt,Y_gt] = meshgrid(-511:1:512);%1024
xdata_gt=zeros(size(X_gt,1),size(X_gt,2),2);
xdata_gt(:, :, 1) = X_gt;
xdata_gt(:, :, 2) = Y_gt;  
input_img=zeros(128,128);
output_img=zeros(1024,1024);

for i =1:5 %15,16,17,18,19,20

sigmax=rand(1)*5.7+5.5;%[75.9-154.56]um
sigmay=rand(1)*4.1+3.9;%[53.82-110.4]um   
while sigmay>sigmax
    sigmay=rand(1)*4.1+3.9;%[2-5] 
end
x_128=randi([-63 64],1);
y_128=randi([-63 64],1);
x_1024=round(x_128*fact_128/fact_1024);
y_1024=round(y_128*fact_128/fact_1024);

input_128=[1, x_128,sigmax,y_128 ,sigmay,0];
Z_input=D2Gauss(input_128,xdata);
input_1024=[1,x_1024,3,y_1024,3,0];%sigmax=5.1um
Z_gt=D2Gauss(input_1024,xdata_gt);

output_img=output_img+Z_gt;
input_img=input_img+Z_input;
x_128=x_128+64;
y_128=y_128+64;
x_1024=x_1024+512;
y_1024=y_1024+512;
rx=[rx x_1024];%based on matlab coordination reference
ry=[ry y_1024];
end

writematrix([rx;ry],'val_5BB.xlsm','WriteMode','append');
filename_label = sprintf('label_img_%d.jpg',j);
cd('C:\Users\Olive\Desktop\Fit2DGaussian Function to Data\Multi_BB\val_#5_outputs');
imwrite(output_img,filename_label);
cd('C:\Users\Olive\Desktop\Fit2DGaussian Function to Data');
input_img=Addnoise(input_img,10);
filename2 = sprintf('x_img_%d.jpg', j);
cd('C:\Users\Olive\Desktop\Fit2DGaussian Function to Data\Multi_BB\val_#5_inputs');
imwrite(input_img,filename2);
cd('C:\Users\Olive\Desktop\Fit2DGaussian Function to Data');

end


%%
% BW=imbinarize(I,1);
% figure()
% imshow(BW)
% impixelregion;
% s = regionprops(BW,'centroid');
% centroids = cat(1,s.Centroid);
% %writematrix(centroids,'20_GT.xls','WriteMode','append');
% hold on
% plot(centroids(:,1),centroids(:,2),'b*')
% hold off
% title('Prediction image')
%%
%filter size does not change too much
%sigma bigger, the center value lower
% BW=BW*1;
% result=imgaussfilt(BW,300,'FilterSize',15); 
% figure()
% imshow(result)
% impixelregion;
%when the bubble is too samll, it would be disappear if sigma is large, 