import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES']='0'  ##gpu
#os.environ['CUDA_VISIBLE_DEVICES'] = "1"   ##cpu
import tensorflow as tf
config = tf.compat.v1.ConfigProto(gpu_options = 
                          tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.4))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)
import re
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras import losses
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from skimage import io
from numpy import newaxis

import math                                  
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
        #tf.math.multiply(7,6)
        loss_1 = losses.mean_squared_error(y_pred,y_true)#*1024*1024
        loss_2= losses.mean_absolute_error(y_pred,tf.zeros(input_shape))
        loss=loss_1 + 0.1*loss_2
        
        return loss
    return bump_mse


#Conv block 1
inputs=tf.keras.Input(shape=(128, 128, 1))#128,128,1
x=layers.Conv2D(16, 3, strides=1, padding='same',activation=None, name="Conv_1")(inputs)#128,128,16
x=layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha=0.3)(x)
conca0=x
x= layers.Conv2D(16, 3, strides=1,padding='same', activation=None, name="Conv_2")(x)#128,128,16
x= layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha=0.3)(x)
x=tf.keras.layers.concatenate([x, conca0])
x= layers.MaxPool2D(pool_size=(2, 2))(x)#64,64,16
conca1=x

 #Conv block 2
x=layers.Conv2D(32, 3, strides=1,padding='same', activation=None, name="Conv_3")(x)#64,64,32
x=layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha=0.3)(x)
x=tf.keras.layers.concatenate([x, conca1])
x= layers.Conv2D(32, 3, strides=1,padding='same', activation=None, name="Conv_4")(x)#64,64,32
x= layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha=0.3)(x)
x=tf.keras.layers.concatenate([x, conca1])
x= layers.MaxPool2D(pool_size=(2, 2))(x)#32,32,32
conca2=x

#Conv block 3
x=layers.Conv2D(64, 3, strides=1,padding='same', activation=None, name="Conv_5")(x)#32,32,64
x= layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha=0.3)(x)
x=tf.keras.layers.concatenate([x, conca2])
x= layers.Conv2D(64, 3, strides=1,padding='same', activation=None, name="Conv_6")(x)
x= layers.BatchNormalization()(x)
x= layers.LeakyReLU(alpha=0.3)(x)
x=tf.keras.layers.concatenate([x, conca2])
x= layers.MaxPool2D(pool_size=(2, 2))(x)#16,16,64
conca3=x

 #latent block #no any leak
x= layers.Conv2D(128, 3, strides=1, padding='same',activation="relu", name="Conv_7")(x)#16,16,128
x= layers.Dropout(0.5)(x)
x= layers.Conv2D(128, 3, strides=1, padding='same',activation="relu", name="Conv_8")(x)

 #Deconv block 1
x=layers.Conv2DTranspose(64,5,strides=1,padding='same',activation="relu",name="DeConv_1")(x)#16,16,64
x=tf.keras.layers.concatenate([x, conca3])
x= layers.Conv2DTranspose(64,5,strides=2,padding='same',activation="relu",name="DeConv_2")(x)#32,32,64
x=tf.keras.layers.concatenate([x, conca2])
x= layers.UpSampling2D(size=(2, 2))(x)#64,64,64
x=tf.keras.layers.concatenate([x, conca1])

 #Deconv block 2
x= layers.Conv2DTranspose(32,5,strides=1,padding='same',activation="relu",name="DeConv_3")(x)#64,64,32
x=tf.keras.layers.concatenate([x, conca1])####
x= layers.Conv2DTranspose(32,5,strides=2,padding='same',activation="relu",name="DeConv_4")(x)#128,128,32
x=tf.keras.layers.concatenate([x, conca0])
x= layers.UpSampling2D(size=(2, 2))(x)#256,256,32

 #Deconv block 3
x= layers.Conv2DTranspose(16,5,strides=1,padding='same',activation="relu",name="DeConv_5")(x)#256,256,16
x= layers.Conv2DTranspose(16,5,strides=2,padding='same',activation="relu",name="DeConv_6")(x)#512,512,16
x= layers.UpSampling2D(size=(2, 2))(x)#1024,1024,16
outputs= layers.Conv2D(1, 3, strides=1, padding='same',activation="linear",#1024,1024,1
           kernel_initializer="Orthogonal",name="Conv_9")(x)

initial_model=tf.keras.Model(inputs=inputs,outputs=outputs)
initial_model.summary()

plot_model(initial_model, to_file='mUNet.png', show_shapes=True)

x_train=[]
for root, dirnames, filenames in os.walk(r"C:\Fit2DGaussian Function to Data\Single_BB\10db_circle_inputs"):
    for filename in filenames:
        if re.search("\.(jpg|jpeg|JPEG|png|PNG|JPG)$", filename):
            filepath = os.path.join(root, filename)
           # pdb.set_trace()
            image = plt.imread(filepath)
            image=image.astype('float32')
            #image,_,_ = project_01(image)
            x_train.append(image)
x_train=np.array(x_train)

x_train = x_train.reshape(x_train.shape[0], 128, 128, 1)
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])


    
y_train=[]
for root, dirnames, filenames in os.walk(r'C:\Fit2DGaussian Function to Data\Single_BB\10db_circle_outputs'): 
    for filename in filenames:
        if re.search("\.(jpg|jpeg|JPEG|png|PNG|JPG)$", filename):
            filepath = os.path.join(root, filename)
            image = plt.imread(filepath)
            image=image.astype('float32')
            y_train.append(image)
y_train=np.array(y_train)
y_train = y_train.reshape(y_train.shape[0], 1024, 1024, 1)


### cross entropy ##
X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.2,
                                                    random_state=42, shuffle=True,)

# Inform user training begun
print('Training model...')


# Save the model weights after each epoch if the validation loss decreased
checkpointer = ModelCheckpoint(
    filepath=r"C:\Fit2DGaussian Function to Data\Single_BB\circleU", 
                               verbose=1, save_best_only=True)
# loss history recorder
history = LossHistory()
# Change learning when loss reaches a plataeu
change_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=0.0000005)

opt = tf.keras.optimizers.Adam(learning_rate=0.0001)   
initial_model.compile(optimizer=opt, loss = L1L2loss((1024, 1024, 1),(7,7),1))
train_history=initial_model.fit(x=X_train,y=Y_train,batch_size=1, epochs=60, verbose=1,
                                callbacks=[history,checkpointer,change_lr],\
                                validation_data=(X_test, Y_test) ) 


loss = train_history.history['loss']
val_loss = train_history.history['val_loss']
total_loss,total_val_loss=[],[]
total_loss+=loss
total_val_loss+=val_loss





print('Training Completed!')


plt.figure()
plt.plot(total_loss)
plt.plot(total_val_loss)
plt.legend(['train_loss', 'val_loss'])
plt.xlabel('Iteration #')
plt.ylabel('Loss Function')
plt.title("Loss function progress during training")
plt.show()
print('training process!')   

# keras_model_path = r"C:\Fit2DGaussian Function to Data\Multi_BB\Model_U_20db_#5"
# initial_model.save(keras_model_path) 


#                                       ############## Evaluation patch image size ##############

# # # load the saved model 
# initial_model = tf.keras.models.load_model(r'C:\Users\Olive\Desktop\Fit2DGaussian Function to Data\Single_BB\Model_Unet_sig3',compile=False)
# initial_model.compile(optimizer='adam', loss = L1L2loss((1024, 1024, 1),(7,7),1))
# initial_model.summary()
# train_history=model.fit(x=X_train,y=Y_train,batch_size=1, epochs=50,verbose=1,
#                                 callbacks=[history,checkpointer,change_lr],\
#                                 validation_data=(X_test, Y_test) ) 
# total_loss,total_val_loss=[],[]        
# loss = train_history.history['loss']
# val_loss = train_history.history['val_loss']
# total_loss+=loss
# total_val_loss+=val_loss


# import pdb
# pdb.set_trace()
pred_im=[]
path = r'C:\Users\Olive\Desktop\Ucircle'
I=getitem()
#pdb.set_trace()
I=I.astype('float32')
for i in range(I.shape[0]):#I.shape[0]
    res=np.zeros([1024,1024])
    for j in range(1):
    #im,mini,maxi = project_01(I[i, :, :])#necessary
        im=I[i,:,:] 
        im=im.reshape(1,128,128,1)
        im=initial_model.predict(im)
        im=im.reshape(1024,1024)
        im[im < 0] = 0
        #im[im==np.max(im)] = 255
        #im[im<np.max(im)] = 0
        #maxi,mini=np.max(im),np.min(im)
        #im=(maxi-mini)*im+mini
        #pred_im.append(im)
        
        # plt.figure()
        # plt.imshow(im,cmap='gray')
        # #plt.axis('off')
        # cv.imwrite(os.path.join(path , "frame_%d.jpg" % i), im)
        res=res+im
    plt.figure()
    plt.imshow(res,cmap='gray')
    cv.imwrite(os.path.join(path , 'U_%d.jpg' % (i+1)), res)

        #     sample=np.squeeze(I)
        #     patch,patch_size=cutting_image(sample,fact=128,overlap=0.5)
        
np.max(res)
np.min(res)
        
np.max(im)
np.min(im)
#im=(maxi-mini)*im+mini
# plt.figure()
# plt.imshow(X_train[3,:,:,:],cmap='gray')
# plt.figure()
# plt.imshow(Y_train[3,:,:,:],cmap='gray')
# cv.imwrite(os.path.join(path , 'pure.jpg'), Y_train[3,:,:,:])

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





