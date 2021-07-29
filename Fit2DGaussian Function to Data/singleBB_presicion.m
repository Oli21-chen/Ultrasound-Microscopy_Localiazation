% Center_1024=(512,512)
cd('C:\Users\Olive\Desktop\Fit2DGaussian Function to Data');
x=[];
y=[];
for a=0:49
%     cd('C:\Users\Olive\Desktop\10db_#1U_precision');
%     filename = ['10db_#1U_' num2str(a,'%2d') '.jpg'];
   cd('C:\Users\Olive\Desktop\10db_#1H_precision');
   filename = ['10db_#1H_' num2str(a,'%2d') '.jpg'];
   img = imread(filename);
   BW=imbinarize(img);
   s = regionprops(BW,'centroid');
   centroids = cat(1,s.Centroid);
   x=[x centroids(1)];
   y=[y centroids(2)];
%    figure()
%    imshow(img)
end
figure()
plot(x-512,y-512,'*b')
hold on
plot(0,0,'or')
hold off
cd('C:\Users\Olive\Desktop\Fit2DGaussian Function to Data');
