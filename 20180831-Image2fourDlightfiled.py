# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 09:13:41 2018

@author: think
"""
"""
Image compression via tensor decomposition
==========================================
Example on how to use :func:`tensorly.decomposition.parafac`and :func:`tensorly.decomposition.tucker` on images.
"""

import matplotlib.pyplot as plt
import tensorly as tl
import numpy as np
from scipy.misc import face, imresize, imread
from tensorly.decomposition import parafac
from tensorly.decomposition import tucker
from tensorly.decomposition import non_negative_parafac
from tensorly.base import unfold, fold
from math import ceil

res = 80 # define resolution of the image **must be devided by 4**
random_state = 12521
# load image data sets
image11 = tl.tensor(imresize(imread('11.png'), (res,res)), dtype='float16')
image12 = tl.tensor(imresize(imread('12.png'), (res,res)), dtype='float16')
image13 = tl.tensor(imresize(imread('13.png'), (res,res)), dtype='float16')
image14 = tl.tensor(imresize(imread('14.png'), (res,res)), dtype='float16')
image15 = tl.tensor(imresize(imread('15.png'), (res,res)), dtype='float16')
image16 = tl.tensor(imresize(imread('16.png'), (res,res)), dtype='float16')
image17 = tl.tensor(imresize(imread('17.png'), (res,res)), dtype='float16')
image18 = tl.tensor(imresize(imread('18.png'), (res,res)), dtype='float16')
image19 = tl.tensor(imresize(imread('19.png'), (res,res)), dtype='float16')
image21 = tl.tensor(imresize(imread('21.png'), (res,res)), dtype='float16')
image22 = tl.tensor(imresize(imread('22.png'), (res,res)), dtype='float16')
image23 = tl.tensor(imresize(imread('23.png'), (res,res)), dtype='float16')
image24 = tl.tensor(imresize(imread('24.png'), (res,res)), dtype='float16')
image25 = tl.tensor(imresize(imread('25.png'), (res,res)), dtype='float16')
image26 = tl.tensor(imresize(imread('26.png'), (res,res)), dtype='float16')
image27 = tl.tensor(imresize(imread('27.png'), (res,res)), dtype='float16')
image28 = tl.tensor(imresize(imread('28.png'), (res,res)), dtype='float16')
image29 = tl.tensor(imresize(imread('29.png'), (res,res)), dtype='float16')
image31 = tl.tensor(imresize(imread('31.png'), (res,res)), dtype='float16')
image32 = tl.tensor(imresize(imread('32.png'), (res,res)), dtype='float16')
image33 = tl.tensor(imresize(imread('33.png'), (res,res)), dtype='float16')
image34 = tl.tensor(imresize(imread('34.png'), (res,res)), dtype='float16')
image35 = tl.tensor(imresize(imread('35.png'), (res,res)), dtype='float16')
image36 = tl.tensor(imresize(imread('36.png'), (res,res)), dtype='float16')
image37 = tl.tensor(imresize(imread('37.png'), (res,res)), dtype='float16')
image38 = tl.tensor(imresize(imread('38.png'), (res,res)), dtype='float16')
image39 = tl.tensor(imresize(imread('39.png'), (res,res)), dtype='float16')
image41 = tl.tensor(imresize(imread('41.png'), (res,res)), dtype='float16')
image42 = tl.tensor(imresize(imread('42.png'), (res,res)), dtype='float16')
image43 = tl.tensor(imresize(imread('43.png'), (res,res)), dtype='float16')
image44 = tl.tensor(imresize(imread('44.png'), (res,res)), dtype='float16')
image45 = tl.tensor(imresize(imread('45.png'), (res,res)), dtype='float16')
image46 = tl.tensor(imresize(imread('46.png'), (res,res)), dtype='float16')
image47 = tl.tensor(imresize(imread('47.png'), (res,res)), dtype='float16')
image48 = tl.tensor(imresize(imread('48.png'), (res,res)), dtype='float16')
image49 = tl.tensor(imresize(imread('49.png'), (res,res)), dtype='float16')
image51 = tl.tensor(imresize(imread('51.png'), (res,res)), dtype='float16')
image52 = tl.tensor(imresize(imread('52.png'), (res,res)), dtype='float16')
image53 = tl.tensor(imresize(imread('53.png'), (res,res)), dtype='float16')
image54 = tl.tensor(imresize(imread('54.png'), (res,res)), dtype='float16')
image55 = tl.tensor(imresize(imread('55.png'), (res,res)), dtype='float16')
image56 = tl.tensor(imresize(imread('56.png'), (res,res)), dtype='float16')
image57 = tl.tensor(imresize(imread('57.png'), (res,res)), dtype='float16')
image58 = tl.tensor(imresize(imread('58.png'), (res,res)), dtype='float16')
image59 = tl.tensor(imresize(imread('59.png'), (res,res)), dtype='float16')
image61 = tl.tensor(imresize(imread('61.png'), (res,res)), dtype='float16')
image62 = tl.tensor(imresize(imread('62.png'), (res,res)), dtype='float16')
image63 = tl.tensor(imresize(imread('63.png'), (res,res)), dtype='float16')
image64 = tl.tensor(imresize(imread('64.png'), (res,res)), dtype='float16')
image65 = tl.tensor(imresize(imread('65.png'), (res,res)), dtype='float16')
image66 = tl.tensor(imresize(imread('66.png'), (res,res)), dtype='float16')
image67 = tl.tensor(imresize(imread('67.png'), (res,res)), dtype='float16')
image68 = tl.tensor(imresize(imread('68.png'), (res,res)), dtype='float16')
image69 = tl.tensor(imresize(imread('69.png'), (res,res)), dtype='float16')
image71 = tl.tensor(imresize(imread('71.png'), (res,res)), dtype='float16')
image72 = tl.tensor(imresize(imread('72.png'), (res,res)), dtype='float16')
image73 = tl.tensor(imresize(imread('73.png'), (res,res)), dtype='float16')
image74 = tl.tensor(imresize(imread('74.png'), (res,res)), dtype='float16')
image75 = tl.tensor(imresize(imread('75.png'), (res,res)), dtype='float16')
image76 = tl.tensor(imresize(imread('76.png'), (res,res)), dtype='float16')
image77 = tl.tensor(imresize(imread('77.png'), (res,res)), dtype='float16')
image78 = tl.tensor(imresize(imread('78.png'), (res,res)), dtype='float16')
image79 = tl.tensor(imresize(imread('79.png'), (res,res)), dtype='float16')
image81 = tl.tensor(imresize(imread('81.png'), (res,res)), dtype='float16')
image82 = tl.tensor(imresize(imread('82.png'), (res,res)), dtype='float16')
image83 = tl.tensor(imresize(imread('83.png'), (res,res)), dtype='float16')
image84 = tl.tensor(imresize(imread('84.png'), (res,res)), dtype='float16')
image85 = tl.tensor(imresize(imread('85.png'), (res,res)), dtype='float16')
image86 = tl.tensor(imresize(imread('86.png'), (res,res)), dtype='float16')
image87 = tl.tensor(imresize(imread('87.png'), (res,res)), dtype='float16')
image88 = tl.tensor(imresize(imread('88.png'), (res,res)), dtype='float16')
image89 = tl.tensor(imresize(imread('89.png'), (res,res)), dtype='float16')
image91 = tl.tensor(imresize(imread('91.png'), (res,res)), dtype='float16')
image92 = tl.tensor(imresize(imread('92.png'), (res,res)), dtype='float16')
image93 = tl.tensor(imresize(imread('93.png'), (res,res)), dtype='float16')
image94 = tl.tensor(imresize(imread('94.png'), (res,res)), dtype='float16')
image95 = tl.tensor(imresize(imread('95.png'), (res,res)), dtype='float16')
image96 = tl.tensor(imresize(imread('96.png'), (res,res)), dtype='float16')
image97 = tl.tensor(imresize(imread('97.png'), (res,res)), dtype='float16')
image98 = tl.tensor(imresize(imread('98.png'), (res,res)), dtype='float16')
image99 = tl.tensor(imresize(imread('99.png'), (res,res)), dtype='float16')

#check image data shape
print('\n****************')
print(image99.shape)
print('****************','\n','*Load image OK**')
print('\n****************')
'''
#create new 4d tensor
#Initial_X = tl.tensor(np.arange(4294967296).reshape((256, 256, 256, 256)))
#orignal_shape = Initial_X.shape
#X = Initial_X
#print (X.shape)  #show shape of image
#print (X)       #image is RGB255 chanels matrix (230, 307, 3)
'''
# write image matrix to 4d tensor
imageset = [image11, image12, image13, image14, image15,image16, image17, image18, image19,
            image21, image22, image23, image24, image25,image26, image27, image28, image29,
            image31, image32, image33, image34, image35,image36, image37, image38, image39,
            image41, image42, image43, image44, image45,image46, image47, image48, image49,
            image51, image52, image53, image54, image55,image56, image57, image58, image59,
            image61, image62, image63, image64, image65,image66, image67, image68, image69,
            image71, image72, image73, image74, image75,image76, image77, image78, image79,
            image81, image82, image83, image84, image85,image86, image87, image88, image89,
            image91, image92, image93, image94, image95,image96, image97, image98, image99]
 
def image_to_tensor(imageset):
    #imageset=[image11,...,image55]
    X = []
    #u, v, s, t, c = X.shape
    for pick in imageset:        
        X.append(pick)
    return X 

def to_image(tensor):
    """A convenience function to convert from a float dtype back to uint8"""
    im = tl.to_numpy(tensor)
    im -= 0
    im /= im.max()
    im *= 255
    return im.astype(np.uint8)

# X = tl.tensor(np.array(image_to_tensor(imageset)),dtype='float32')
#Initial_X is a five dimention Tensor, u,v,s,t,c   C is the chanel of RGB
Initial_X = np.reshape(np.array(image_to_tensor(imageset)), (9,9,res,res,3))
X = Initial_X
print('\n\n****************','\n')
print('Initial_X is done')
print('****************')
#extend X in u dimention 
step = int(res / 8) # step x 8 = resolution
sp = 0
insert = []
out =[]
# 1H extend
frontslice = X[0,:,:,:,:].astype('float16')
rearslice = X[1,:,:,:,:].astype('float16')
for sp in range(0,step):
    '''if sp < step/2:
        insert = frontslice
    else:
        insert = rearslice''' 
    insert = (1 - sp/step)*frontslice +  (sp/step)*rearslice
    insert = insert[np.newaxis,:,:,:,:]
    out = np.vstack([X,insert])
    X = out#[[0,5,1,2,3,4],:,:,:,:]
    print('u of stpe',sp)
#2H extend
frontslice = X[1,:,:,:,:].astype('float16')
rearslice = X[2,:,:,:,:].astype('float16')
for sp in range(0,step):
    '''if sp < step/2:
        insert = frontslice
    else:
        insert = rearslice'''
    insert = (1 - sp/step)*frontslice +  (sp/step)*rearslice 
    insert = insert[np.newaxis,:,:,:,:]
    out = np.vstack([X,insert])
    X = out#[[0,5,1,2,3,4],:,:,:,:]
    print('u of stpe',sp + step )
# 3H extend
frontslice = X[2,:,:,:,:].astype('float16')
rearslice = X[3,:,:,:,:].astype('float16')
for sp in range(0,step):
    '''if sp < step/2:
        insert = frontslice
    else:
        insert = rearslice''' 
    insert = (1 - sp/step)*frontslice +  (sp/step)*rearslice 
    insert = insert[np.newaxis,:,:,:,:]
    out = np.vstack([X,insert])
    X = out#[[0,5,1,2,3,4],:,:,:,:]
    print('u of stpe',sp + step*2)
#4H extend
frontslice = X[3,:,:,:,:].astype('float16')
rearslice = X[4,:,:,:,:].astype('float16')
for sp in range(0,step):
    '''if sp < step/2:
        insert = frontslice
    else:
        insert = rearslice''' 
    insert = (1 - sp/step)*frontslice +  (sp/step)*rearslice 
    insert = insert[np.newaxis,:,:,:,:]
    out = np.vstack([X,insert])
    X = out#[[0,5,1,2,3,4],:,:,:,:]
    print('u of stpe',sp + step*3)
# 5H extend
frontslice = X[4,:,:,:,:].astype('float16')
rearslice = X[5,:,:,:,:].astype('float16')
for sp in range(0,step):
    '''if sp < step/2:
        insert = frontslice
    else:
        insert = rearslice''' 
    insert = (1 - sp/step)*frontslice +  (sp/step)*rearslice
    insert = insert[np.newaxis,:,:,:,:]
    out = np.vstack([X,insert])
    X = out#[[0,5,1,2,3,4],:,:,:,:]
    print('u of stpe',sp + step*4)
#6H extend
frontslice = X[5,:,:,:,:].astype('float16')
rearslice = X[6,:,:,:,:].astype('float16')
for sp in range(0,step):
    '''if sp < step/2:
        insert = frontslice
    else:
        insert = rearslice'''
    insert = (1 - sp/step)*frontslice +  (sp/step)*rearslice 
    insert = insert[np.newaxis,:,:,:,:]
    out = np.vstack([X,insert])
    X = out#[[0,5,1,2,3,4],:,:,:,:]
    print('u of stpe',sp + step*5 )
# 7H extend
frontslice = X[6,:,:,:,:].astype('float16')
rearslice = X[7,:,:,:,:].astype('float16')
for sp in range(0,step):
    '''if sp < step/2:
        insert = frontslice
    else:
        insert = rearslice''' 
    insert = (1 - sp/step)*frontslice +  (sp/step)*rearslice 
    insert = insert[np.newaxis,:,:,:,:]
    out = np.vstack([X,insert])
    X = out#[[0,5,1,2,3,4],:,:,:,:]
    print('u of stpe',sp + step*6)
#8H extend
frontslice = X[7,:,:,:,:].astype('float16')
rearslice = X[8,:,:,:,:].astype('float16')
for sp in range(0,step):
    '''if sp < step/2:
        insert = frontslice
    else:
        insert = rearslice''' 
    insert = (1 - sp/step)*frontslice +  (sp/step)*rearslice 
    insert = insert[np.newaxis,:,:,:,:]
    out = np.vstack([X,insert])
    X = out#[[0,5,1,2,3,4],:,:,:,:]
    print('u of stpe',sp + step*7)
    
#delete the source images
X = np.delete(X,[0,1,2,3,4,5,6,7,8],axis=0)
print('Insert Tensor is of shape',X.shape)

#######################
#extend X in v dimention
step = int(res / 8)
sp = 0
insert = []
out =[]
#1H extend
frontslice = X[:,0,:,:,:].astype('float16')
rearslice = X[:,1,:,:,:].astype('float16')
for sp in range(0,step):
    '''if sp < step/2:
        insert = frontslice
    else:
        insert = rearslice'''
    insert = (1 - sp/step)*frontslice +  (sp/step)*rearslice 
    insert = insert[:,np.newaxis,:,:,:]
    out = np.hstack([X,insert])
    X = out#[[0,5,1,2,3,4],:,:,:,:]
    print('v of stpe',sp)
#2H extend
frontslice = X[:,1,:,:,:].astype('float16')
rearslice = X[:,2,:,:,:].astype('float16')
for sp in range(0,step):
    '''if sp < step/2:
        insert = frontslice
    else:
        insert = rearslice'''
    insert = (1 - sp/step)*frontslice +  (sp/step)*rearslice 
    insert = insert[:,np.newaxis,:,:,:]
    out = np.hstack([X,insert])
    X = out#[[0,5,1,2,3,4],:,:,:,:]
    print('v of stpe',sp + step)
#3H extend
frontslice = X[:,2,:,:,:].astype('float16')
rearslice = X[:,3,:,:,:].astype('float16')
for sp in range(0,step):
    '''if sp < step/2:
        insert = frontslice
    else:
        insert = rearslice''' 
    insert = (1 - sp/step)*frontslice +  (sp/step)*rearslice 
    insert = insert[:,np.newaxis,:,:,:]
    out = np.hstack([X,insert])
    X = out#[[0,5,1,2,3,4],:,:,:,:]
    print('v of stpe',sp + step*2)
#4H extend
frontslice = X[:,3,:,:,:].astype('float16')
rearslice = X[:,4,:,:,:].astype('float16')
for sp in range(0,step):
    '''if sp < step/2:
        insert = frontslice
    else:
        insert = rearslice''' 
    insert = (1 - sp/step)*frontslice +  (sp/step)*rearslice 
    insert = insert[:,np.newaxis,:,:,:]
    out = np.hstack([X,insert])
    X = out#[[0,5,1,2,3,4],:,:,:,:]
    print('v of stpe',sp + step*3)
#5H extend
frontslice = X[:,4,:,:,:].astype('float16')
rearslice = X[:,5,:,:,:].astype('float16')
for sp in range(0,step):
    '''if sp < step/2:
        insert = frontslice
    else:
        insert = rearslice'''
    insert = (1 - sp/step)*frontslice +  (sp/step)*rearslice 
    insert = insert[:,np.newaxis,:,:,:]
    out = np.hstack([X,insert])
    X = out#[[0,5,1,2,3,4],:,:,:,:]
    print('v of stpe',sp + step*4)
#6H extend
frontslice = X[:,5,:,:,:].astype('float16')
rearslice = X[:,6,:,:,:].astype('float16')
for sp in range(0,step):
    '''if sp < step/2:
        insert = frontslice
    else:
        insert = rearslice'''
    insert = (1 - sp/step)*frontslice +  (sp/step)*rearslice 
    insert = insert[:,np.newaxis,:,:,:]
    out = np.hstack([X,insert])
    X = out#[[0,5,1,2,3,4],:,:,:,:]
    print('v of stpe',sp + step*5)
#7H extend
frontslice = X[:,6,:,:,:].astype('float16')
rearslice = X[:,7,:,:,:].astype('float16')
for sp in range(0,step):
    '''if sp < step/2:
        insert = frontslice
    else:
        insert = rearslice '''
    insert = (1 - sp/step)*frontslice +  (sp/step)*rearslice 
    insert = insert[:,np.newaxis,:,:,:]
    out = np.hstack([X,insert])
    X = out#[[0,5,1,2,3,4],:,:,:,:]
    print('v of stpe',sp + step*6)
#8H extend
frontslice = X[:,7,:,:,:].astype('float16')
rearslice = X[:,8,:,:,:].astype('float16')
for sp in range(0,step):
    '''if sp < step/2:
        insert = frontslice
    else:
        insert = rearslice''' 
    insert = (1 - sp/step)*frontslice +  (sp/step)*rearslice 
    insert = insert[:,np.newaxis,:,:,:]
    out = np.hstack([X,insert])
    X = out#[[0,5,1,2,3,4],:,:,:,:]
    print('v of stpe',sp + step*7)
#########################
#delete the source images
X = np.delete(X,[0,1,2,3,4,5,6,7,8],axis=1)
        #print('u=',u,';\n\n sp=',sp)
print('\n\n****************','\n')
print('Insert Tensor is of shape',X.shape)
print('****************','/n/n/n','*******************')
#save 4d light fileds 
#change the name correcte
np.save('80x80_BOE 4dlightfield',X) 
print('\n\n****************','\n')
print('lightfield has been saved successfully!!')
print('****************','\n\n')

'''
#Load the saved 4d light filed npy
X = []
X = np.load('100x100 4dlightfiled.npy')
print('################','\n','4d lightfiled load successfully!','\n\n','################')

X = tl.tensor(X, dtype = 'float64') # change the dtype of X 
#RGB THREE CHANNEL CP Decomposition separately
R_X = X[:, :, :, :, 0]
G_X = X[:, :, :, :, 1]
B_X = X[:, :, :, :, 2]
# Rank of the CP decomposition
rank = 1
# Rank of the Tucker decomposition
#tucker_rank = [100, 100, 2]

# Perform the CP decomposition
#factors = parafac(X, rank=cp_rank, init='random', tol=10e-3)
nnfactors = non_negative_parafac(R_X, rank = rank, n_iter_max = 30, init = 'random',tol = 10e-6)
lcd1 = nnfactors[0]
lcd2 = nnfactors[1]
lcd3 = nnfactors[2]
lcd4 = nnfactors[3]
print('\n\n****************','\n')
print('NTF Decomposition complite !!')
print(lcd1.shape, lcd2.shape, lcd3.shape, lcd4.shape)
print(lcd1.dtype, lcd2.dtype, lcd3.dtype, lcd4.dtype)
print('****************','\n\n')

# Reconstruct the image from the factors
cp_reconstruction = tl.kruskal_to_tensor(nnfactors)
print(cp_reconstruction.shape)

# Tucker decomposition
#core, tucker_factors = tucker(image, ranks=tucker_rank, init='random', tol=10e-5, random_state=random_state)
#tucker_reconstruction = tl.tucker_to_tensor(core, tucker_factors)

##fram_rank_reconstruction = tl.tensor(np.arange(307200).reshape((640, 480)))
##for i in range(1,rank):

#fram_rank_reconstruction = np.matrix(lcd1)* np.matrix(lcd2).T
#print(fram_rank_reconstruction)
##rank_reconstruction= fram_rank_reconstruction / rank
# Plotting the original and reconstruction from the decompositions
'''
fig = plt.figure()
ax = fig.add_subplot(2, 3, 1)
ax.set_axis_off()
ax.imshow(to_image(X[22, 22, : , : , :]))
ax.set_title('11 image')

ax = fig.add_subplot(2, 3, 2)
ax.set_axis_off()
ax.imshow(to_image(X[25, 22,: , : , :]))
ax.set_title('12')

ax = fig.add_subplot(2, 3, 3)
ax.set_axis_off()
ax.imshow(to_image(X[35, 35, : , : , :]))
ax.set_title('13')


ax = fig.add_subplot(2, 3, 4)
ax.set_axis_off()
ax.imshow(to_image(X[22, :, : , 22 , :]))
ax.set_title('14')

ax = fig.add_subplot(2, 3, 5)
ax.set_axis_off()
ax.imshow(to_image(X[56, :, 22 , : , :]))
ax.set_title('15')


plt.tight_layout()
plt.show()
