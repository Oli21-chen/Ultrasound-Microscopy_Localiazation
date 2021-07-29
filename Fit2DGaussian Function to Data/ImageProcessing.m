 I=imread("IMG_001.jpg");
 %[R,G,B]=imsplit(I);
 %gs=im2gray(I);
 gs=imread("gs.jpg");
 %%observe the hist intensity
 %imhist(gs)
 %%enhence the contrast,
 %Darker pixels will shift to lower bins, while brighter pixels will shift to higher bins
 %gs2Adj=imadjust(gs);
 %imshowpair(gs,gs2Adj,"montage")
 %%imlocalbrighten to adjust the contrast of a color image.
 %I2adj=imlocalbrighten(I)
 
 %%imtool 图像调整

 %%segement
 %BW=gs2Adj>150;
 %imshow(BW)
 
 %%
 %To automate the threshold selection process, you can use the imbinarize function, which calculates the "best" threshold for the image.
 %得到的结果用可以sum(),判断每一行/列有什么东西
 %BW=imbinarize(gsAdj,"adaptive","ForegroundPolarity","dark");
 %imshow(BW)
 %% %% %%
 %pre- and postprocessing to improve segmentation
 %%Noise Removal
 %Smooth pixel intensity values to reduce the impact of variation on binarization.
 %%Background Isolation and Subtraction
 %Isolate and remove the background of an image before binarizing.
 %%Binary Morphology
 %Emphasize particular patterns or shapes in a binary image. 
 %%
 %%Use the fspecial function to create an n-by-n averaging filter.
 %F = fspecial("average",n)
 %%You can apply a filter F to an image I by using the imfilter function.
 %Ifltr = imfilter(I,F,"replicate");%Instead of zeros, the "replicate" option uses pixel intensity values on the image border for pixels outside the image.
 %%
 %%Remove background subtraction
 %Structuring elements are created using the strel function.
 %SE = strel("diamond",5)
 %%To perform a closing operation on an image I with structuring element SE, use the imclose function.
 %Iclosed = imclose(I,SE);%enhance bright area
 %then use Iclosed-I to derive the inverse diagram, then use
 %imbinarize(),~inverse the matrix
 %%above all,use BW=imbothat(gs,SE)
 
 %imopen(I,SE);%enhance dark area
 %montage({I,BW,BWstripes}) % Show the image

 %% normalization
 B = rescale(A,0,255); % Normalizes image to [0 255]
B = rescale(A); % Normalizes image (0 to 1, by default)
%% 
I= imread('data1.jpg');
grayImage = rgb2gray(I);
subplot(1,2,1);
imshow(grayImage);
axis on;
ft = fftshift(log(abs(fft2(grayImage))));
subplot(1,2,2);
imshow(ft, []);
axis on;
 
 %% 改变像素数量，但不改变真是图像保真度
 J = imresize(I,scale);
 
 
 