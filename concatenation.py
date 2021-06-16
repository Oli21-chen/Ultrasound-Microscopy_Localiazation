import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES']='0'
import matplotlib.pyplot as plt
import numpy as np
from cuttingimage import getitem,cutting_image
from skimage.transform import rescale
   

        
fact=150
super_patch=[]
I=getitem()#np.array
for I_i in range (I.shape[0]):
    im=I[I_i,:,:]
    plt.figure()
    plt.imshow(im,cmap='gray')

    
    patches,sz=cutting_image(im,fact=150)
    row_num=int(sz[0]/fact)
    col_num=int(sz[1]/fact)
    
    for pat_i in range (patches.shape[2]):
        pat=patches[:,:,pat_i]
        # plt.figure()
        # plt.imshow(pat,cmap='gray')
        '''
        tf.predict, return a image
        #super_resolution_im=initial_model.predict(pat)
        '''
     
        pred_img = rescale(pat, 2, anti_aliasing=False)
        plt.figure()
        plt.imshow(pred_img,cmap='gray')
        super_patch.append(pred_img)
        
  
    sub_patch=[None]*col_num
    sub_patch_col=0
    i=1
    while i <len(super_patch):
        if i==1:
            sub_patch[sub_patch_col]=np.concatenate((super_patch[0], super_patch[i]), axis=0)
            i=i+1
            
        elif (i% row_num!= 0) :
            
            sub_patch[sub_patch_col]=np.concatenate((sub_patch[sub_patch_col], super_patch[i]), axis=0)  
            i=i+1
             
        elif (i% row_num == 0) :
            sub_patch_col=sub_patch_col+1
            i=i+1
            sub_patch[sub_patch_col]=np.concatenate((super_patch[i-1], super_patch[i]), axis=0)
            i=i+1
        
    # plt.figure()
    # plt.imshow(sub_patch[2],cmap='gray')    
    for i in range(1, col_num):
        if i==1:
            result_sr_img=np.concatenate((sub_patch[i-1], sub_patch[i]), axis=1)  
        else: 
            result_sr_img=np.concatenate((result_sr_img, sub_patch[i]), axis=1)  
            
  
    plt.figure()
    plt.imshow(result_sr_img,cmap='gray')   