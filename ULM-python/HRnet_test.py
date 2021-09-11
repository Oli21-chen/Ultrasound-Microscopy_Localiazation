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
        loss_1 = losses.mean_squared_error(y_pred,y_true)#*1024*1024
        loss_2= losses.mean_absolute_error(y_pred,tf.zeros(input_shape))
        loss=loss_1 +1*loss_2
        
        return loss
    return bump_mse


def conv_bn_relu(nb_filter, rk, ck, name):
    def f(input):
        conv = layers.Conv2D(nb_filter, kernel_size=(rk, ck), strides=(1,1),\
                               padding="same", use_bias=False,\
                               kernel_initializer="Orthogonal",name='conv-'+name)(input)
        conv_norm = layers.BatchNormalization(name='BN-'+name)(conv)
        conv_norm_relu = layers.LeakyReLU(alpha=0.3)(conv_norm)
        return conv_norm_relu
    return f
def pool_layer(input,f_num,names):#small
    pool1 = layers.MaxPooling2D(pool_size=(2,2),name=names+'Pool')(input)
    F1 = conv_bn_relu(f_num,3,3,names+'_1')(pool1)
    F2 = conv_bn_relu(f_num,3,3,names+'_2')(F1)
    F3 = conv_bn_relu(f_num,3,3,names+'_3')(F2)
    F4 = conv_bn_relu(f_num,3,3,names+'_4')(F3)
    return F4
def same_layer(input,f_num,names):#same
    F1 = conv_bn_relu(f_num,3,3,names+'_1')(input)
    F2 = conv_bn_relu(f_num,3,3,names+'_2')(F1)
    F3 = conv_bn_relu(f_num,3,3,names+'_3')(F2)
    F4 = conv_bn_relu(f_num,3,3,names+'_4')(F3)
    return F4

inputs=tf.keras.Input(shape=(128, 128, 1))#128,128,1
L=conv_bn_relu(16, 3, 3, 'conv_1')(inputs)
L=conv_bn_relu(16, 3, 3, 'conv_2')(L)
L=conv_bn_relu(16, 3, 3, 'conv_3')(L)#blc_1
#### the same line
M=pool_layer(L,32,'blc_M_2')#blc_2_end
L=same_layer(L,16,'blc_L_2')#blc_2_end

##  transform blc 2-3
L2M = layers.MaxPooling2D(pool_size=(2,2),name='L2M_b23')(L)
L2S=layers.MaxPooling2D(pool_size=(2,2),name='L2S_b23')(L2M)
M2L=layers.UpSampling2D(size=(2, 2),name='M2L_b23')(M)
M2S=layers.MaxPooling2D(pool_size=(2,2),name='M2S_b23')(M)

link_blc2_L=tf.keras.layers.concatenate([L, M2L])
link_blc2_M=tf.keras.layers.concatenate([L2M, M])
link_blc2_S=tf.keras.layers.concatenate([L2S, M2S])
S=same_layer(link_blc2_S,64,'blc_S_3')#blc_3_end
M=same_layer(link_blc2_M,32,'blc_M_3')#blc_3_end
L=same_layer(link_blc2_L,16,'blc_L_3')#blc_3_end

##  transform blc 3-4
L2M = layers.MaxPooling2D(pool_size=(2,2),name='L2M_b34')(L)
L2S=layers.MaxPooling2D(pool_size=(2,2),name='L2S_b34')(L2M)
L2XS=layers.MaxPooling2D(pool_size=(2,2),name='L2XS_b34')(L2S)
M2L=layers.UpSampling2D(size=(2, 2),name='M2L_b34')(M)
M2S=layers.MaxPooling2D(pool_size=(2,2),name='M2S_b34')(M)
M2XS=layers.MaxPooling2D(pool_size=(2,2),name='M2XS_b34')(M2S)
S2M=layers.UpSampling2D(size=(2, 2),name='S2M_b34')(S)
S2L=layers.UpSampling2D(size=(2, 2),name='S2L_b34')(S2M)
S2XS=layers.MaxPooling2D(pool_size=(2,2),name='S2XS_b34')(S)


link_blc3_L=tf.keras.layers.concatenate([L, M2L,S2L])
link_blc3_M=tf.keras.layers.concatenate([M, L2M,S2M])
link_blc3_S=tf.keras.layers.concatenate([S,L2S, M2S])
link_blc3_XS=tf.keras.layers.concatenate([L2XS, M2XS,S2XS])
L=same_layer(link_blc3_L,16,'blc_L_4')#blc_4_end
M=same_layer(link_blc3_M,32,'blc_M_4')#blc_4_end
S=same_layer(link_blc3_S,64,'blc_S_4')#blc_4_end
XS=same_layer(link_blc3_XS,128,'blc_XS_4')#blc_4



## transform together
L2M=layers.MaxPooling2D(pool_size=(2,2),name='L2M_b45')(L)
L2S=layers.MaxPooling2D(pool_size=(2,2),name='L2S_b45')(L2M)
L2XS=layers.MaxPooling2D(pool_size=(2,2),name='L2XS_b45')(L2S)
M2L=layers.UpSampling2D(size=(2, 2),name='M2L_b45')(M)
M2S=layers.MaxPooling2D(pool_size=(2,2),name='M2S_b45')(M)
M2XS=layers.MaxPooling2D(pool_size=(2,2),name='M2XS_b45')(M2S)
S2M=layers.UpSampling2D(size=(2, 2),name='S2M_b45')(S)
S2L=layers.UpSampling2D(size=(2, 2),name='S2L_b45')(S2M)
S2XS=layers.MaxPooling2D(pool_size=(2,2),name='S2XS_b45')(S)
XS2S=layers.UpSampling2D(size=(2, 2),name='XS2S_b45')(XS)
XS2M=layers.UpSampling2D(size=(2, 2),name='XS2M_b45')(XS2S)
XS2L=layers.UpSampling2D(size=(2, 2),name='XS2L_b45')(XS2M)

link_blc4_L=tf.keras.layers.concatenate([L, M2L,S2L,XS2L])
link_blc4_M=tf.keras.layers.concatenate([M, L2M,S2M,XS2M])
link_blc4_S=tf.keras.layers.concatenate([S,L2S, M2S,XS2S])
link_blc4_XS=tf.keras.layers.concatenate([XS,L2XS, M2XS,S2XS])
# =============================================================================
# # L=same_layer(link_blc4_L,16,'blc_L_5')#blc_5_end
# # M=same_layer(link_blc4_M,32,'blc_M_5')#blc_5_end
# # S=same_layer(link_blc4_S,64,'blc_S_5')#blc_5_end
# # XS=same_layer(link_blc4_XS,128,'blc_XS_5')#blc_5_end
# 
# # L2M=layers.MaxPooling2D(pool_size=(2,2),name='L2M_final')(L)
# # L2S=layers.MaxPooling2D(pool_size=(2,2),name='L2S_final')(L2M)
# # L2XS=layers.MaxPooling2D(pool_size=(2,2),name='L2XS_final')(L2S)
# # M2L=layers.UpSampling2D(size=(2, 2),name='M2L_final')(M)
# # M2S=layers.MaxPooling2D(pool_size=(2,2),name='M2S_final')(M)
# # M2XS=layers.MaxPooling2D(pool_size=(2,2),name='M2XS_final')(M2S)
# # S2M=layers.UpSampling2D(size=(2, 2),name='S2M_final')(S)
# # S2L=layers.UpSampling2D(size=(2, 2),name='S2L_final')(S2M)
# # S2XS=layers.MaxPooling2D(pool_size=(2,2),name='S2XS_final')(S)
# # XS2S=layers.UpSampling2D(size=(2, 2),name='XS2S_final')(XS)
# # XS2M=layers.UpSampling2D(size=(2, 2),name='XS2M_final')(XS2S)
# # XS2L=layers.UpSampling2D(size=(2, 2),name='XS2L_final')(XS2M)
# # final_blc=tf.keras.layers.concatenate([XS,M2XS, S2XS,L2XS])
# # x= layers.Dropout(0.5)(final_blc)
# =============================================================================

 #Deconv block 1
x=layers.Conv2DTranspose(64,5,strides=1,padding='same',activation="relu",name="DeConv_1")(link_blc4_XS)#16,16,64
x= layers.Conv2DTranspose(64,5,strides=2,padding='same',activation="relu",name="DeConv_2")(x)#32,32,64
x= layers.UpSampling2D(size=(2, 2))(x)#64,64,64

 #Deconv block 2
x= layers.Conv2DTranspose(32,5,strides=1,padding='same',activation="relu",name="DeConv_3")(x)#64,64,32
x= layers.Conv2DTranspose(32,5,strides=2,padding='same',activation="relu",name="DeConv_4")(x)#128,128,32
x= layers.UpSampling2D(size=(2, 2))(x)#256,256,32

 #Deconv block 3
x= layers.Conv2DTranspose(16,5,strides=1,padding='same',activation="relu",name="DeConv_5")(x)#256,256,16
x= layers.Conv2DTranspose(16,5,strides=2,padding='same',activation="relu",name="DeConv_6")(x)#512,512,16
x= layers.UpSampling2D(size=(2, 2))(x)#1024,1024,16
outputs= layers.Conv2D(1, 3, strides=1, padding='same',activation="linear",#1024,1024,1
           kernel_initializer="Orthogonal",name="Conv_9")(x)
# x= layers.Conv2D(64, 3, strides=1, padding='same',activation="relu", name="Conv_7")(final_blc)
#x=tf.nn.depth_to_space( x, 8, data_format='NHWC', name='subpixel')
# outputs= layers.Conv2D(1, 3, strides=1, padding='same',activation="linear",#1024,1024,1
#             kernel_initializer="Orthogonal",name="Conv_8")(x)
initial_model=tf.keras.Model(inputs=inputs,outputs=outputs)
initial_model.summary()
#plot_model(initial_model, to_file='model_1.png', show_shapes=True)

x_train=[]
for root, dirnames, filenames in os.walk(r"C:\Fit2DGaussian Function to Data\Single_BB\10db_circle_inputs"):
    for filename in filenames:
        if re.search("\.(jpg|jpeg|JPEG|png|PNG|JPG)$", filename):
            filepath = os.path.join(root, filename)
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
checkpointer = ModelCheckpoint(filepath=r"C:\Fit2DGaussian Function to Data\Single_BB\circleH",
                               verbose=1,
                                save_best_only=True)
# loss history recorder
history = LossHistory()
# Change learning when loss reaches a plataeu
change_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=0.0000005)
##################################



opt = tf.keras.optimizers.Adam(learning_rate=0.0001) #0.0001
initial_model.compile(optimizer=opt, loss = L1L2loss((1024, 1024, 1),(7,7),1))

train_history=initial_model.fit(x=X_train,y=Y_train,batch_size=1, epochs=60,verbose=1,
                                callbacks=[history,checkpointer,change_lr],\
                                validation_data=(X_test, Y_test) ) 
total_loss,total_val_loss=[],[]      
loss = train_history.history['loss']
val_loss = train_history.history['val_loss']
total_loss+=loss
total_val_loss+=val_loss


# keras_model_path = r"C:\Users\Olive\Desktop\Fit2DGaussian Function to Data\Single_BB\Noise_Model_HRnet"
# initial_model.save(keras_model_path)  # save() should be called out of strategy scope 
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



#                                       ############## Evaluation patch image size ##############

# # load the saved model 
# model = tf.keras.models.load_model(r'C:\Users\Olive\Desktop\Fit2DGaussian Function to Data\Single_BB\Model_HRnet_sig3',compile=False)
# model.compile(optimizer='adam', loss = L1L2loss((1024, 1024, 1),(7,7),1))
# model.summary()



        
pred_im=[]
path = r'C:\Users\Olive\Desktop\Hcircle'
I=getitem()
#pdb.set_trace()
I=I.astype('float32')


for i in range(I.shape[0]):#I.shape[0]
    res=np.zeros([1024,1024])
    for j in range(1):
        im=I[i,:,:] 
        im=im.reshape(1,128,128,1)
        im=initial_model.predict(im)
        im=im.reshape(1024,1024)
        im[im < 0] = 0
        # plt.figure()
        # plt.imshow(im,cmap='gray')
        # #plt.axis('off')
        # cv.imwrite(os.path.join(path , "frame_%d.jpg" % i), im)
        res=res+im
    # plt.figure()
    # plt.imshow(res,cmap='gray')
    cv.imwrite(os.path.join(path , 'H_%d.jpg' % (i+1)), res)


