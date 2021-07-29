function noise_img= Addnoise(img,snr)
img=im2double(img);
varI=std2(img)^2;
SNRdB=snr;
sigma_noise = sqrt(varI/10^(SNRdB/10));
noise_img=imnoise(img, 'Gaussian', 0, sigma_noise^2);

end
% I = im2double(I);
% varI = std2(I)^2;
% SNRdB = 5:5:30;
% for i=1:numel(SNRdB)
%   sigma_noise = sqrt(varI/10^(SNRdB(i)/10));
%   N = sigma_noise*randn(size(I));
%   IN1 = I+N; % using randn
%   IN2 = imnoise(I, 'Gaussian', 0, sigma_noise^2); % using imnoise
%  imshow([IN1 IN2])
%   title(['SNR = ' int2str(SNRdB(i)) 'dB' ...
%     ', \sigma_{noise} = ' num2str(sigma_noise)]);
%   disp('Press any key to proceed')
%   pause
%end