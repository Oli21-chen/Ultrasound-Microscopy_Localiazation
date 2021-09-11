import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES']='0'  ##gpu
#os.environ['CUDA_VISIBLE_DEVICES'] = "1"   ##cpu
import tensorflow as tf
config = tf.compat.v1.ConfigProto(gpu_options = 
                          tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.3))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

from tensorflow.keras import backend as K
from tensorflow.keras import losses
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
import numpy as np
                                 
from En_cuttingImage_half_overlap import getitem,cutting_image
from En_concatenation_half_overlap import  concanate
import cv2 as cv
import pdb


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        
def normalize_im(im, dmean, dstd):
    im = np.squeeze(im)
    im_norm = np.zeros(im.shape,dtype=np.float32)
    im_norm = (im - dmean)/dstd
    return im_norm

def project_01(im):
    im = np.squeeze(im)
    min_val = im.min()
    max_val = im.max()
    norm_im=(im - min_val)/(max_val - min_val)
    return norm_im,min_val,max_val

#  Define a matlab like gaussian 2D filter
def matlab_style_gauss2D(shape=(7,7),sigma=1):#keep shape samll induce higher pixel value
    """ 
    2D gaussian filter - should give the same result as:
    MATLAB's fspecial('gaussian',[shape],[sigma]) 
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h.astype(dtype=K.floatx())
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    h = h*2.0
    h = h.astype('float32')
    return h

# Combined MSE + L1 loss
def L1L2loss(input_shape,gshape,sig):
    psf_heatmap = matlab_style_gauss2D(shape = gshape,sigma=sig)
    gfilter = tf.reshape(psf_heatmap, [gshape[0], gshape[1], 1, 1])
    
    def bump_mse(y_true, y_pred):
        # generate the heatmap corresponding to the predicted spikes
        y_true = K.conv2d(y_true, gfilter, strides=(1, 1), padding='same', 
                              dilation_rate=(1, 1))
        loss_1 = losses.mean_squared_error(y_pred,y_true)
        loss_2= losses.mean_absolute_error(y_pred,tf.zeros(input_shape))
        loss=loss_1 + 0.1*loss_2
        
        return loss
    return bump_mse


new_model = tf.keras.models.load_model(r'C:\Fit2DGaussian Function to Data\Multi_BB\Model_H_10db_#10',compile=False)
new_model.compile(optimizer='adam', loss = L1L2loss((1024, 1024, 1),(7,7),1))
new_model.summary()



pred_im=[]
path = r'C:\Users\Olive\Desktop\H10db_3px'
I=getitem()
#pdb.set_trace()
I=I.astype('float32')


for i in range(I.shape[0]):#I.shape[0]
    res=np.zeros([1024,1024])
    for j in range(1):
        im=I[i,:,:] 
        im=im.reshape(1,128,128,1)
        im=new_model.predict(im)
        im=im.reshape(1024,1024)
        im[im < 0] = 0
        # plt.figure()
        # plt.imshow(im,cmap='gray')
        # #plt.axis('off')
        # cv.imwrite(os.path.join(path , "frame_%d.jpg" % i), im)
        res=res+im
    # plt.figure()
    # plt.imshow(res,cmap='gray')
    cv.imwrite(os.path.join(path , 'H10db_%d.jpg' % (i+1)), res)

        #     sample=np.squeeze(I)
        #     patch,patch_size=cutting_image(sample,fact=128,overlap=0.5)
        #     sample=np.squeeze(I)
        #     patch,patch_size=cutting_image(sample,fact=128,overlap=0.5)
     


# for ind in range(np.shape(patch)[2]):
#     im=patch[:,:,ind]
#     im=im.reshape(1,128,128,1)
#     im=initial_model.predict(im)
#     im=im.reshape(1024,1024)
#     im[im < 0] = 0
#     im=im*255
#     pred_im.append(im)
#     plt.figure()
#     plt.imshow(im,cmap='gray')
#     cv.imwrite(os.path.join(path , "frame_%d.jpg" % ind), im)
# pred_im=np.array(pred_im)
# pred_im=np.moveaxis(pred_im,0,-1)
# results=concanate(pred_im, patch_size, fact=128, overlap=0.5)


##########
# patches=pred_im
# sz=patch_size
# fact=128
# overlap=0.5
# fact_row, fact_col=8,8#128-1024
# num_row,num_col=sz[0]/fact,sz[1]/fact
# row_num=int(sz[0]/(fact*overlap)-1)############overlap
# #col_num=int(sz[1]/(fact*overlap)-1)################
# stride=int(fact*fact_row/2)
# conca=[[None]]
# conca_row=[]
# conca_result=[]
# i=0
# switch_index=[j for j in range(np.shape(patches)[2]) if (j)%row_num==0]
# #while np.shape(conca)[0]<fact*3:
# while i<np.shape(patches)[2]:#super_patch = patch[2]
#     if i in switch_index:
#         first_up=patches[:,:,i][:stride,:]
#         first_down=patches[:,:,i][stride:,:]
#         i=i+1
#         second_up=patches[:,:,i][:stride,:]
#         second_down=patches[:,:,i][stride:,:]
#         #latent=(first_down+second_up)/2
#         latent=np.maximum(first_down, second_up)
#         conca=np.concatenate((first_up,latent),axis=0)
#         # plt.figure()
#         # plt.imshow(conca,cmap='gray')
#     else:
#         i=i+1
#         first_up=second_up
#         first_down=second_down
#         second_up=patches[:,:,i][:stride,:]
#         second_down=patches[:,:,i][stride:,:]
#         #latent=(first_down+second_up)/2   ### Average
#         latent=np.maximum(first_down, second_up)  ## Max value
#         conca=np.concatenate((conca,latent),axis=0)
#         # plt.figure()
#         # plt.imshow(conca,cmap='gray')
#     if np.shape(conca)[0]==(fact*fact_row*num_row-stride):#*放大倍数，*几个new patch
#         conca=np.concatenate((conca,second_down),axis=0)
#         i=i+1
#         plt.figure()
#         plt.imshow(conca,cmap='gray')
#         conca_row.append(conca)
#         conca=[[None]]
        
# i=0
# while i<len(conca_row):
#     if i ==0:
#         first_left=conca_row[i][:,:stride]
#         first_right=conca_row[i][:,stride:]
#         i=i+1
#         second_left=conca_row[i][:,:stride]
#         second_right=conca_row[i][:,stride:]
#         #latent=(first_right+second_left)/2  ### Average
#         latent=np.maximum(first_right, second_left)  ## Max value
#         conca=np.concatenate((first_left,latent),axis=1)
#         # plt.figure()
#         # plt.imshow(conca,cmap='gray')
#     else:
#         i=i+1
#         first_left=second_left
#         first_right=second_right
#         second_left=conca_row[i][:,:stride]
#         second_right=conca_row[i][:,stride:]
#         #latent=(first_right+second_left)/2 ### Average
#         latent=np.maximum(first_right, second_left) ## Max value
#         conca=np.concatenate((conca,latent),axis=1)
#         # plt.figure()
#         # plt.imshow(conca,cmap='gray')
#     if np.shape(conca)[1]==fact*fact_col*num_col-stride:
#         conca=np.concatenate((conca,second_right),axis=1)
#         i=i+1
#         plt.figure()
#         plt.imshow(conca,cmap='gray')
#         conca_result.append(conca)

        


# plt.figure()
# plt.imshow(im,cmap='gray')
# cv.imwrite(os.path.join(path , "result.jpg"), results)

