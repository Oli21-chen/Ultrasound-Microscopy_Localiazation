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
# from En_cuttingImage_half_overlap import getitem,cutting_image
# from En_concatenation_half_overlap import  concanate
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


def _conv_bn_relu(filter_number, ks,stride_conv, name):
    def f(input):
        conv = layers.Conv2D(filter_number, kernel_size=ks, strides=stride_conv,activation=None,
                               padding="same", use_bias=False,name=name+'conv-')(input)
        conv_norm = layers.BatchNormalization(momentum=0.1,name=name+'BN-')(conv)
        conv_norm_relu = layers.ReLU()(conv_norm)
        return conv_norm_relu
    return f


def _forward_path(input,filter_number,ks,stride_conv,names):#small
    x = _conv_bn_relu(filter_number,ks,stride_conv,names+'_1')(input)
    x = _conv_bn_relu(filter_number,ks,stride_conv,names+'_2')(x)
    x = _conv_bn_relu(filter_number,ks,stride_conv,names+'_3')(x)
    x = _conv_bn_relu(filter_number,ks,stride_conv,names+'_4')(x)
    return x
def _image_downsampling(input,filter_number,ks,stride_conv,names):
    #x = layers.MaxPooling2D(pool_size=p_size,strides=stride_pool,name=names+'Pooling')(input)
    x = _conv_bn_relu(filter_number,ks,stride_conv,names+'Pool_')(input)
    return x
def _image_upsampling(input,sz,filter_number,ks,stride_conv,names):
    x = layers.UpSampling2D(sz,interpolation='nearest',name=names+'Upsampling')(input)
    x = _conv_bn_relu(filter_number,ks,stride_conv,names+'Upsample_')(x)
    return x
def _image_self(input,filter_number,ks,stride_conv,names):
    x = _conv_bn_relu(filter_number,ks,stride_conv,names+'_Unchanging')(input)
    return x
#initialte
C=64
#### Block 1 ####
inputs=tf.keras.Input(shape=(128, 128, 1),name='Input')#128,128,1
b1_L = _forward_path(inputs,filter_number=C, ks=3, stride_conv=1, names='B1')
b1_L_= _image_self(b1_L,filter_number=C,ks=3,stride_conv=1,names='B1')
b1_M_ = _image_downsampling(b1_L,filter_number=2*C,ks=3,stride_conv=2,names='B1')

#### Block 2 ####
b2_L = _forward_path(b1_L_,filter_number=C, ks=3, stride_conv=1, names='B2_L')
b2_M = _forward_path(b1_M_,filter_number=2*C, ks=3, stride_conv=1, names='B2_M')
#b2_L_ include L M
b2_M2L = _image_upsampling(b2_M,sz=2,filter_number=C,ks=1,stride_conv=1,names='B2_M2L')
b2_L2L = _image_self(b2_L,filter_number=C,ks=3,stride_conv=1,names='B2_L2L')
b2_L_= tf.keras.layers.concatenate([b2_L2L, b2_M2L])#128,128,128
#b2_M_ include L M
b2_M2M = _image_self(b2_M,filter_number=2*C,ks=3,stride_conv=1,names='B2_M2M')
b2_L2M = _image_downsampling(b2_L,filter_number=2*C,ks=3,stride_conv=2,names='B2_L2M')
b2_M_ = tf.keras.layers.concatenate([b2_M2M, b2_L2M])#64,64,256
#b2_S_ include L M
b2_M2S = _image_downsampling(b2_M,filter_number=4*C,ks=3,stride_conv=2,names='B2_M2S')
b2_L2S = _image_downsampling(b2_L2M,filter_number=4*C,ks=3,stride_conv=2,names='B2_L2S')
b2_S_ = tf.keras.layers.concatenate([b2_M2S, b2_L2S])#32, 32, 512

#### Block 3 ####
b3_L =  _forward_path(b2_L_,filter_number=C,ks=3,stride_conv=1,names='B3_L')
b3_M =  _forward_path(b2_M_,filter_number=2*C,ks=3,stride_conv=1,names='B3_M')
b3_S =  _forward_path(b2_S_,filter_number=4*C,ks=3,stride_conv=1,names='B3_S')
#b3_L_ include L M S
b3_L2L = _image_self(b3_L,filter_number=C,ks=3,stride_conv=1,names='B3_L2L')
b3_M2L = _image_upsampling(b3_M,sz=2,filter_number=C,ks=1,stride_conv=1,names='B3_M2L')
b3_S2L = _image_upsampling(b3_S,sz=4,filter_number=C,ks=1,stride_conv=1,names='B3_S2L')
b3_L_ = tf.keras.layers.concatenate([b3_L2L, b3_M2L,b3_S2L])#128, 128, 192
#b3_M_ include L M S
b3_L2M = _image_downsampling(b3_L,filter_number=2*C,ks=3,stride_conv=2,names='B3_L2M')
b3_M2M = _image_self(b3_M,filter_number=2*C,ks=3,stride_conv=1,names='B3_M2M')
b3_S2M = _image_upsampling(b3_S,sz=2,filter_number=2*C,ks=1,stride_conv=1,names='B3_S2M')#upsample 1*1 conv
b3_M_ = tf.keras.layers.concatenate([b3_L2M, b3_M2M,b3_S2M])#64, 64, 384
#b3_S_ include L M S
b3_L2S = _image_downsampling(b3_L2M,filter_number=4*C,ks=3,stride_conv=2,names='B3_L2S')
b3_M2S = _image_downsampling(b3_M,filter_number=4*C,ks=3,stride_conv=2,names='B3_M2S')
b3_S2S =  _image_self(b3_S,filter_number=4*C,ks=3,stride_conv=1,names='B3_S2S')
b3_S_ = tf.keras.layers.concatenate([b3_L2S, b3_M2S,b3_S2S])#32,32,768
#b3_XS_ include LMS
b3_L2XS = _image_downsampling(b3_L2S,filter_number=8*C,ks=3,stride_conv=2,names='B3_L2XS')
b3_M2XS = _image_downsampling(b3_M2S,filter_number=8*C,ks=3,stride_conv=2,names='B3_M2XS')
b3_S2XS = _image_downsampling(b3_S,filter_number=8*C,ks=3,stride_conv=2,names='B3_S2XS')
b3_XS_ = tf.keras.layers.concatenate([b3_L2XS,b3_M2XS,b3_S2XS])# 16, 16, 1536

#### block 4 ####
b4_L = _forward_path(b3_L_,filter_number=C,ks=3,stride_conv=1,names='B4_L')
b4_M = _forward_path(b3_M_,filter_number=2*C,ks=3,stride_conv=1,names='B4_M')
b4_S = _forward_path(b3_S_,filter_number=4*C,ks=3,stride_conv=1,names='B4_S')
b4_XS = _forward_path(b3_XS_,filter_number=8*C,ks=3,stride_conv=1,names='B4_XS')
#b4_L_
b4_L2L = _image_self(b4_L,filter_number=C,ks=3,stride_conv=1,names='B4_L2L')
b4_M2L = _image_upsampling(b4_M,sz=2,filter_number=C,ks=1,stride_conv=1,names='B4_M2L')
b4_S2L = _image_upsampling(b4_S,sz=4,filter_number=C,ks=1,stride_conv=1,names='B4_S2L')
b4_XS2L =  _image_upsampling(b4_XS,sz=8,filter_number=C,ks=1,stride_conv=1,names='B4_XS2L')
b4_L_ = tf.keras.layers.concatenate([b4_L2L,b4_M2L,b4_S2L,b4_XS2L])# 128, 128, 256
#b4_M_
b4_L2M = _image_downsampling(b4_L,filter_number=2*C,ks=3,stride_conv=2,names='B4_L2M')
b4_M2M = _image_self(b4_M,filter_number=2*C,ks=3,stride_conv=1,names='B4_M2M')
b4_S2M = _image_upsampling(b4_S,sz=2,filter_number=2*C,ks=1,stride_conv=1,names='B4_S2M')
b4_XS2M = _image_upsampling(b4_XS,sz=4,filter_number=2*C,ks=1,stride_conv=1,names='B4_XS2M')
b4_M_ = tf.keras.layers.concatenate([b4_L2M,b4_M2M,b4_S2M,b4_XS2M])#  64, 64, 512
#b4_S_
b4_L2S = _image_downsampling(b4_L2M,filter_number=4*C,ks=3,stride_conv=2,names='B4_L2S')
b4_M2S = _image_downsampling(b4_M,filter_number=4*C,ks=3,stride_conv=2,names='B4_M2S')
b4_S2S = _image_self(b4_S,filter_number=4*C,ks=3,stride_conv=1,names='B4_S2S')
b4_XS2S = _image_upsampling(b4_XS,sz=2,filter_number=4*C,ks=1,stride_conv=1,names='B4_XS2S')
b4_S_ = tf.keras.layers.concatenate([b4_L2S,b4_M2S,b4_S2S,b4_XS2S])#  32, 32, 1024
#b4_XS_
b4_L2XS = _image_downsampling(b4_L2S,filter_number=8*C,ks=3,stride_conv=2,names='B4_L2XS')
b4_M2XS = _image_downsampling(b4_M2S,filter_number=8*C,ks=3,stride_conv=2,names='B4_M2XS')
b4_S2XS = _image_downsampling(b4_S,filter_number=8*C,ks=3,stride_conv=2,names='B4_S2XS')
b4_XS2XS = _image_self(b4_XS,filter_number=8*C,ks=3,stride_conv=1,names='B4_XS2XS')
b4_XS_ = tf.keras.layers.concatenate([b4_L2XS,b4_M2XS,b4_S2XS,b4_XS2XS])#  16, 16, 2048

#### block fusion to L ####
# head_XS2L = _image_upsampling(b4_XS_,sz=8,filter_number=C,ks=1,stride_conv=1,names='head_XS2L')
# head_S2L =  _image_upsampling(b4_S_,sz=4,filter_number=C,ks=1,stride_conv=1,names='head_S2L')
# head_M2L =  _image_upsampling(b4_M,sz=2,filter_number=C,ks=1,stride_conv=1,names='head_M2L')
# head_L2L =  _image_self(b4_L_,filter_number=C,ks=3,stride_conv=1,names='head_L2L')
# head_L  = tf.keras.layers.concatenate([head_L2L,head_M2L,head_S2L,head_XS2L])#  128, 128, 256 
#### block fusion to XS ####
head_L2M_ = _image_downsampling(b4_L_,filter_number=8*C,ks=3,stride_conv=2,names='head_L2M')
head_L2S_ = _image_downsampling(head_L2M_,filter_number=8*C,ks=3,stride_conv=2,names='head_L2S')
head_L2XS = _image_downsampling(head_L2S_,filter_number=8*C,ks=3,stride_conv=2,names='head_L2XS')
head_M2S_ = _image_downsampling(b4_M_,filter_number=8*C,ks=3,stride_conv=2,names='head_M2S')
head_M2XS = _image_downsampling(head_M2S_,filter_number=8*C,ks=3,stride_conv=2,names='head_M2XS')
head_S2XS = _image_downsampling(b4_S_,filter_number=8*C,ks=3,stride_conv=2,names='head_S2XS')
head_XS2XS = _image_self(b4_XS_,filter_number=8*C,ks=3,stride_conv=1,names='head_XS2XS')
head_= tf.keras.layers.concatenate([head_L2XS,head_M2XS,head_S2XS,head_XS2XS])#  16, 16, 2048


  #Deconv block 1
x=layers.Conv2DTranspose(64,3,strides=1,padding='same',activation="relu",name="DeConv_1")(head_)#16,16,64
x= layers.Conv2DTranspose(64,3,strides=2,padding='same',activation="relu",name="DeConv_2")(x)#32,32,64
x= layers.UpSampling2D(size=(2, 2))(x)#64,64,64

  #Deconv block 2
x= layers.Conv2DTranspose(32,3,strides=1,padding='same',activation="relu",name="DeConv_3")(x)#64,64,32
x= layers.Conv2DTranspose(32,3,strides=2,padding='same',activation="relu",name="DeConv_4")(x)#128,128,32
x= layers.UpSampling2D(size=(2, 2))(x)#256,256,32

  #Deconv block 3
x= layers.Conv2DTranspose(16,3,strides=1,padding='same',activation="relu",name="DeConv_5")(x)#256,256,16
x= layers.Conv2DTranspose(16,3,strides=2,padding='same',activation="relu",name="DeConv_6")(x)#512,512,16
x= layers.UpSampling2D(size=(2, 2))(x)#1024,1024,16
x = layers.Conv2D(1, 3, strides=1, padding='same',activation="linear",#1024,1024,1
            kernel_initializer="Orthogonal",name="output_img")(x)


initial_model=tf.keras.Model(inputs=inputs,outputs=x)
initial_model.summary()
#plot_model(initial_model, to_file=r'C:\Users\Olive\Desktop\HRNet.png', show_shapes=True)
#%%
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


