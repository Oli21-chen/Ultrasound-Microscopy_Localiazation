import os
import re
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
# =============================================================================
# cutting image
# Imperial College London
# Dong Chen
# 2021/6/13
# =============================================================================


def getitem():
    I=[]
    for root, dirnames, filenames in os.walk(r'C:\Users\Olive\Desktop\10_10db_in'):
        for i in range(1,np.size(filenames)+1):
            for filename in filenames:
                if re.fullmatch('x_img_%d.jpg' % i, filename):
                    filepath = os.path.join(root, filename)
                    image = plt.imread(filepath)
                    gray_im = rgb2gray(image)
                    #gray_im=gray_im/255
                    I.append(gray_im)
    I=np.array(I)        
    return I



def cutting_image(I,fact=128,overlap=0.5):
    sz=np.shape(I)
    rem=[sz_element % fact for sz_element in sz]
    
    times=[]
    for (item1, item2) in zip (sz,rem):
        times.append((item1-item2)/fact)
    #####  row
    if times[0]==0:
        if rem[0]!=0:
            I=np.append(I,np.zeros((fact-sz[0],sz[1])),axis=0)
            sz=list(np.shape(I))
    else:
        if rem[0]!=0:
            I=np.append(I,np.zeros((fact-(sz[0]-int(times[0])*fact),sz[1])),axis=0)
            sz=list(np.shape(I))
    ######    coloum
    if times[1]==0:
        if rem[1]!=0:
            I=np.append(I , np.zeros((sz[0],fact-sz[1])),axis=1)
            sz=list(np.shape(I))
    else:
        if rem[1]!=0:
            I=np.append(I,np.zeros((sz[0],fact-(sz[1]-int(times[1])*fact))),axis=1)
            sz=list(np.shape(I))     
    # plt.figure()
    # plt.imshow(I)
    
    ####   patches####
    rem=[sz_element % fact for sz_element in sz]
    times=[]
    for (item1, item2) in zip (sz,rem):
        times.append((item1-item2)/(fact*overlap)-1)
    num=np.prod(times)#production of elements in array
    patch=np.zeros((fact,fact,int(num)))
    row=0
    col=0
    
    
    for i in range(int(num)):
        patch[:,:,i]=I[row:row+fact,col:col+fact]
        if row+fact>=sz[0]:
            col=col+int(fact*overlap)
            row=0
        else:
            row=row+int(fact*overlap)
    return patch,sz