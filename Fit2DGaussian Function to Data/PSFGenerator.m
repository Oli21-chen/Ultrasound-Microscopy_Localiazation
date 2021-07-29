%% Generate PSF dataset with groundture data file
% Ultrasound Localization Microscopy through deep learning
% Dong Chen 2021
% Imperial Colloge London
%% This is to generate presetting data with assuming real length 1.7664e+3;%um

real_sz=1.7664e+3;%um
fact_128=real_sz/128;
fact_1024=real_sz/1024;
for i=1:50
sigmax=rand(1)*5.7+5.5;%[75.9-154.56]um
sigmay=rand(1)*4.1+3.9;%[53.82-110.4]um   
while sigmay>sigmax
    sigmay=rand(1)*4.1+3.9;%[2-5] 
end
x_128=0;%randi([-63 64],1);
y_128=0;%randi([-63 64],1);

[X,Y] = meshgrid(-63:1:64);
xdata=zeros(size(X,1),size(X,2),2);
xdata(:, :, 1) = X;
xdata(:, :, 2) = Y; 
input_128=[1, x_128,sigmax,y_128 ,sigmay,0];
Z_input=D2Gauss(input_128,xdata);
Z_noise=Addnoise(Z_input,20);
%  figure()
%  imshow(Z_noise)
% impixelregion

x_1024=round(x_128*fact_128/fact_1024);
y_1024=round(y_128*fact_128/fact_1024);
[X_gt,Y_gt] = meshgrid(-511:1:512);%1024
xdata_gt=zeros(size(X_gt,1),size(X_gt,2),2);
xdata_gt(:, :, 1) = X_gt;
xdata_gt(:, :, 2) = Y_gt;  
input_1024=[1,x_1024,3,y_1024,3,0];%sigmax=5.1um
Z_gt=D2Gauss(input_1024,xdata_gt);
% figure()
% imshow(Z_gt)
%impixelregion

x_128=x_128+64;
y_128=y_128+64;
x_1024=x_1024+512;
y_1024=y_1024+512;

%writematrix([x_1024 y_1024],'single_val_20db.xlsm','WriteMode','append');
filename_label = sprintf('label_img_%d.jpg',i);%('Pure_label.jpg');%
cd('C:\Users\Olive\Desktop\Fit2DGaussian Function to Data\Single_BB\10db_precision_outputs');
imwrite(Z_gt,filename_label);
cd('C:\Users\Olive\Desktop\Fit2DGaussian Function to Data');

cd('C:\Users\Olive\Desktop\Fit2DGaussian Function to Data\Single_BB\10db_precision_inputs');
filename_input = sprintf('img_%d.jpg',i);%('Pure.jpg');%;
imwrite(Z_noise,filename_input);
cd('C:\Users\Olive\Desktop\Fit2DGaussian Function to Data');
end


