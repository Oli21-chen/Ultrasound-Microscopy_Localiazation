I=label_img_1;
x=[1,0,1,0,1,0];
[X,Y] = meshgrid(-50:1:50);
xdata=zeros(size(X,1),size(X,2),2);
xdata(:, :, 1) = X;
xdata(:, :, 2) = Y;
F = D2GaussFunctionRot(x,xdata);

figure
imshow(F,'InitialMagnification',800);

C=conv2(I,F,'same');
figure
imshow(C)