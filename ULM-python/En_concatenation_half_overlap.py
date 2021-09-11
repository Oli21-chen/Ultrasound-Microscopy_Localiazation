import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES']='0'
import matplotlib.pyplot as plt
import numpy as np
from En_cuttingImage_half_overlap import getitem,cutting_image
from skimage.transform import rescale
   

# overlap=0.5     
# fact=128
# super_patch=[]
# I=getitem()#np.array
# for I_i in range (I.shape[0]):
#     im=I[I_i,:,:]
#     plt.figure()
#     plt.imshow(im,cmap='gray')

    
#     patches,sz=cutting_image(im,fact=128)
#     row_num=int(sz[0]/(fact*overlap)-1)############overlap
#     col_num=int(sz[1]/(fact*overlap)-1)################
    
#     for pat_i in range (patches.shape[2]):
#         pat=patches[:,:,pat_i]
#         # plt.figure()
#         # plt.imshow(pat,cmap='gray')
#         '''
#         tf.predict, return a image
#         #super_resolution_im=initial_model.predict(pat)
#         '''
     
#         pred_img = rescale(pat, 2, anti_aliasing=False)
#         plt.figure()
#         plt.imshow(pred_img,cmap='gray')
#         super_patch.append(pred_img)
# patches=np.array(super_patch)
# patches=np.moveaxis(patches,0,-1)


def concanate( patches, sz, fact=128, overlap=0.5):
    fact_row, fact_col=8,8#128-1024
    num_row,num_col=sz[0]/fact,sz[1]/fact
    row_num=int(sz[0]/(fact*overlap)-1)############overlap
    #col_num=int(sz[1]/(fact*overlap)-1)################
    stride=int(fact*fact_row/2)
    conca=[[None]]
    conca_row=[]
    conca_result=[]
    i=0
    switch_index=[j for j in range(np.shape(patches)[2]) if (j)%row_num==0]
    #while np.shape(conca)[0]<fact*3:
    while i<np.shape(patches)[2]:#super_patch = patch[2]
        if i in switch_index:
            first_up=patches[:,:,i][:stride,:]
            first_down=patches[:,:,i][stride:,:]
            i=i+1
            second_up=patches[:,:,i][:stride,:]
            second_down=patches[:,:,i][stride:,:]
            #latent=(first_down+second_up)/2
            latent=np.maximum(first_down, second_up)
            conca=np.concatenate((first_up,latent),axis=0)
            # plt.figure()
            # plt.imshow(conca,cmap='gray')
        else:
            i=i+1
            first_up=second_up
            first_down=second_down
            second_up=patches[:,:,i][:stride,:]
            second_down=patches[:,:,i][stride:,:]
            #latent=(first_down+second_up)/2   ### Average
            latent=np.maximum(first_down, second_up)  ## Max value
            conca=np.concatenate((conca,latent),axis=0)
            # plt.figure()
            # plt.imshow(conca,cmap='gray')
        if np.shape(conca)[0]==(fact*fact_row*num_row-stride):#*放大倍数，*几个new patch
            conca=np.concatenate((conca,second_down),axis=0)
            i=i+1
            plt.figure()
            plt.imshow(conca,cmap='gray')
            conca_row.append(conca)
            conca=[[None]]
            
    i=0
    while i<len(conca_row):
        if i ==0:
            first_left=conca_row[i][:,:stride]
            first_right=conca_row[i][:,stride:]
            i=i+1
            second_left=conca_row[i][:,:stride]
            second_right=conca_row[i][:,stride:]
            #latent=(first_right+second_left)/2  ### Average
            latent=np.maximum(first_right, second_left)  ## Max value
            conca=np.concatenate((first_left,latent),axis=1)
            # plt.figure()
            # plt.imshow(conca,cmap='gray')
        else:
            i=i+1
            first_left=second_left
            first_right=second_right
            second_left=conca_row[i][:,:stride]
            second_right=conca_row[i][:,stride:]
            #latent=(first_right+second_left)/2 ### Average
            latent=np.maximum(first_right, second_left) ## Max value
            conca=np.concatenate((conca,latent),axis=1)
            # plt.figure()
            # plt.imshow(conca,cmap='gray')
        if np.shape(conca)[1]==fact*fact_col*num_col-stride:
            conca=np.concatenate((conca,second_right),axis=1)
            i=i+1
            plt.figure()
            plt.imshow(conca,cmap='gray')
            conca_result.append(conca)
    conca_result=np.array(conca_result)
    conca_result=np.squeeze(conca_result)
    return conca_result   
                
            