import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES']='0'
import tensorflow as tf
config = tf.compat.v1.ConfigProto(gpu_options = 
                          tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.3))
#config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)
import re
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
# from keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint


import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from skimage import io
import time
from numpy import newaxis




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
    return (im - min_val)/(max_val - min_val)

#  Define a matlab like gaussian 2D filter
def matlab_style_gauss2D(shape=(7,7),sigma=1):
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




# Expand the filter dimensions


# Combined MSE + L1 loss
def L1L2loss(input_shape,gshape,sig):
    psf_heatmap = matlab_style_gauss2D(shape = gshape,sigma=sig)
    gfilter = tf.reshape(psf_heatmap, [gshape[0], gshape[1], 1, 1])
    
    def bump_mse(y_true, y_pre):

        # generate the heatmap corresponding to the predicted spikes
        groundtrue = K.conv2d(y_true, gfilter, strides=(1, 1), padding='same')

        # heatmaps MSE
      #  loss_1 = losses.mean_squared_error(y_pre,groundtrue)
        loss_1 = losses.mean_squared_error(y_pre,groundtrue)*1024*1024
        loss_2= losses.mean_absolute_error(y_pre,tf.zeros(input_shape))
      #  loss_2=tf.reduce_sum(losses.mean_absolute_error(y_pre,tf.zeros(input_shape)))
        loss=loss_1 + 0.01*loss_2
        return loss
    return bump_mse

initial_model = tf.keras.Sequential(
    [
        keras.Input(shape=(128, 128, 1)),
        layers.Conv2D(16, 3, strides=1, padding='same',activation=None, name="Conv_1"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.3),
        layers.Conv2D(16, 3, strides=1,padding='same', activation=None, name="Conv_2"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.3),
        layers.MaxPool2D(pool_size=(2, 2)),
        
        layers.Conv2D(32, 3, strides=1,padding='same', activation=None, name="Conv_3"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.3),
        layers.Conv2D(32, 3, strides=1,padding='same', activation=None, name="Conv_4"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.3),
        layers.MaxPool2D(pool_size=(2, 2)),
        
        layers.Conv2D(64, 3, strides=1,padding='same', activation=None, name="Conv_5"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.3),
        layers.Conv2D(64, 3, strides=1,padding='same', activation=None, name="Conv_6"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.3),
        layers.MaxPool2D(pool_size=(2, 2)),
        
        #latent block
        layers.Conv2D(128, 3, strides=1, padding='same',activation="relu", name="Conv_7"),
        layers.Dropout(0.5),
        layers.Conv2D(128, 3, strides=1, padding='same',activation="relu", name="Conv_8"),
        
        
         #Deconv block
        layers.Conv2DTranspose(64,5,strides=1,padding='same',activation="relu",name="DeConv_1"),
       # layers.UpSampling2D(size=(2, 2)),
        layers.Conv2DTranspose(64,5,strides=2,padding='same',activation="relu",name="DeConv_2"),
        layers.UpSampling2D(size=(2, 2)),
        
        layers.Conv2DTranspose(32,5,strides=1,padding='same',activation="relu",name="DeConv_3"),
        layers.Conv2DTranspose(32,5,strides=2,padding='same',activation="relu",name="DeConv_4"),
        layers.UpSampling2D(size=(2, 2)),
        
        
        layers.Conv2DTranspose(16,5,strides=1,padding='same',activation="relu",name="DeConv_5"),
        layers.Conv2DTranspose(16,5,strides=2,padding='same',activation="relu",name="DeConv_6"),
        layers.UpSampling2D(size=(2, 2)),
        layers.Conv2D(1, 5, strides=1, padding='same',activation="linear",\
                      kernel_initializer="Orthogonal",\
                      name="Conv_9")
    ]
)


initial_model.summary()

x_train=[]
#C:/Users/Olive/FYP/Fit2DGaussian Function to Data/X_dataset
#C:/Users/Olive/Desktop/Fit2DGaussian Function to Data/X_dataset
for root, dirnames, filenames in os.walk("C:/Users/Olive/Desktop/Fit2DGaussian Function to Data/X_dataset"):
#for root, dirnames, filenames in os.walk("C:/Users/Olive/FYP/Fit2DGaussian Function to Data/X_dataset"):
    for filename in filenames:
        if re.search("\.(jpg|jpeg|JPEG|png|bmp|tiff)$", filename):
            #if batch_nb == max_batches: 
            #    return x_train_n2, x_train_down2
            filepath = os.path.join(root, filename)
            image = plt.imread(filepath)
            #if len(image.shape) > 2:
                        
            image=image/255
            x_train.append(image)
x_train=np.array(x_train)

#plt.imshow(x_train[1,:,:])

'''
#===================== Training set normalization ==========================
# normalize training images to be in the range [0,1] and calculate the 
# training set mean and std
mean_train = np.zeros(x_train.shape[0],dtype=np.float32)
std_train = np.zeros(x_train.shape[0], dtype=np.float32)
for i in range(x_train.shape[0]):
    x_train[i, :, :] = project_01(x_train[i, :, :])
    mean_train[i] = x_train[i, :, :].mean()
    std_train[i] = x_train[i, :, :].std()

# resulting normalized training images
mean_val_train = mean_train.mean()
std_val_train = std_train.mean()
x_train_norm = np.zeros(x_train.shape, dtype=np.float32)
for i in range(x_train.shape[0]):
    x_train_norm[i, :, :] = normalize_im(x_train[i, :, :], mean_val_train, std_val_train)

plt.imshow(x_train_norm[4,:,:])
# patch size
psize =  x_train_norm.shape[1]

# Reshaping
x_train_norm = x_train_norm.reshape(x_train.shape[0], psize, psize, 1)

plt.imshow(x_train[0], cmap='gray', vmin=0, vmax=255)
'''

x_train = x_train.reshape(x_train.shape[0], 128, 128, 1)
x_train = x_train.astype('float32')

print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])

y_train=[]
#C:/Users/Olive/FYP/Fit2DGaussian Function to Data/Label
#C:/Users/Olive/Desktop/Fit2DGaussian Function to Data/Multi_Label
for root, dirnames, filenames in os.walk("C:/Users/Olive/Desktop/Fit2DGaussian Function to Data/Multi_Label"):
#for root, dirnames, filenames in os.walk("C:/Users/Olive/FYP/Fit2DGaussian Function to Data/Label"):
    for filename in filenames:
        if re.search("\.(jpg|jpeg|JPEG|png|bmp|tiff)$", filename):
            #if batch_nb == max_batches: 
            #    return x_train_n2, x_train_down2
            filepath = os.path.join(root, filename)
            image = plt.imread(filepath)
            #if len(image.shape) > 2:
                        
            #    image_resized = resize(image, (128, 128))
            y_train.append(image)
y_train=np.array(y_train)
y_train = y_train.astype('float32')
y_train /= 255
#plt.imshow(y_train[1,:,:])
y_train = y_train.reshape(y_train.shape[0], 1024, 1024, 1)


X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=42)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
Y_train = Y_train.astype('float32')
Y_test = Y_test.astype('float32')
   
# Save the model weights after each epoch if the validation loss decreased
checkpointer = ModelCheckpoint(filepath="C:/Users/Olive/FYP/Fit2DGaussian Function to Data", verbose=1,
                                save_best_only=True)

# loss history recorder
history = LossHistory()
# Change learning when loss reaches a plataeu
change_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00005)
'''
# have to define manually a dict to store all epochs scores 
history = {}
history['history'] = {}
history['history']['loss'] = []
history['history']['val_loss'] = []

   # Inform user training begun
print('Training model...')
#opt = optimizers.Adam(lr=0.001)


initial_model.compile(optimizer='adam', loss = L1L2loss((1024, 1024, 1),(7,7),1))
# define number of iterations in training and test
batch_size=1
train_iter = round(X_train.shape[0]/batch_size)
test_iter = round(X_test.shape[0]/batch_size)

for epoch in range(5):
    
    # train iterations 
    loss=0
    for i in range(train_iter):
        
        start = i*batch_size
        end = i*batch_size + batch_size
        batchX = X_train[start:end,]
        batchy = Y_train[start:end,]
        
        loss_ = initial_model.train_on_batch(batchX,batchy)
                
        loss += loss_
     
    history['history']['loss'].append(loss/train_iter)
    print('loss:',history['history']['loss'][epoch])
    
    
    # test iterations 
    val_loss= 0
    for i in range(test_iter):
        
        start = i*batch_size
        end = i*batch_size + batch_size
        batchX = X_test[start:end,]
        batchy = Y_test[start:end,]
        
        val_loss_= initial_model.test_on_batch(batchX,batchy)
    
    history['history']['val_loss'].append(val_loss/test_iter)
    print('val_loss:',history['history']['val_loss'][epoch])
 
initial_model.compile(optimizer='adam', loss = L1L2loss((1024, 1024, 1),(21,21),1))

for epoch in range(5):
    
    # train iterations 
    loss=0
    for i in range(train_iter):
        
        start = i*batch_size
        end = i*batch_size + batch_size
        batchX = X_train[start:end,]
        batchy = Y_train[start:end,]
    
        loss_= initial_model.train_on_batch(batchX,batchy)
        
        loss += loss_
      
        
    history['history']['loss'].append(loss/train_iter)
    print('loss2:',history['history']['loss'][epoch])
   
    # test iterations 
    val_loss= 0
    for i in range(test_iter):
        
        start = i*batch_size
        end = i*batch_size + batch_size
        batchX = X_test[start:end,]
        batchy =Y_test[start:end,]
        
        val_loss_ = initial_model.test_on_batch(batchX,batchy)
        
        val_loss += val_loss_
       
        
    history['history']['val_loss'].append(val_loss/test_iter)
    print('val_loss2',history['history']['val_loss'][epoch])
 
plt.title('Loss')
plt.plot(history['history']['loss'], label='train')
plt.plot(history['history']['val_loss'], label='test')
plt.legend()
'''
initial_model.compile(optimizer='adam', loss = L1L2loss((1024, 1024, 1),(7,7),1))
train_history=initial_model.fit(x=X_train,y=Y_train,batch_size=1, epochs=50 ,verbose=1,callbacks=[history,checkpointer,change_lr],\
                  validation_data=(X_test, Y_test) ) 

loss = train_history.history['loss']
val_loss = train_history.history['val_loss']

total_loss,total_val_loss=[],[]
total_loss+=loss
total_val_loss+=val_loss
'''
initial_model.compile(optimizer='adam', loss = L1L2loss((1024, 1024, 1),(7,7),1))

train_history=initial_model.fit(x=X_train,y=Y_train,batch_size=1, epochs=25,verbose=1,callbacks=[history,checkpointer,change_lr],\
                  validation_data=(X_test, Y_test) ) 
    
loss = train_history.history['loss']
val_loss = train_history.history['val_loss']
total_loss+=loss
total_val_loss+=val_loss
'''
print('Training Completed!')


plt.figure()
plt.plot(total_loss)
plt.plot(total_val_loss)
plt.legend(['train_loss', 'val_loss'])
plt.xlabel('Iteration #')
plt.ylabel('Loss Function')
plt.title("Loss function progress during training")
plt.show()



print('Finish!')   


Images = io.imread('x_img_300.jpg')
Images = Images[:, :, newaxis]
plt.figure()
plt.imshow(Images)
Images=Images.reshape(1,128,128,1)
norImage=Images/ 255

pred_img= initial_model.predict(norImage)

pred_img=pred_img.reshape(1024,1024,1)
pred_img[pred_img < 0] = 0
plt.figure()
plt.imshow(pred_img)
print('Finish!')
