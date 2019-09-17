# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 17:28:46 2018

@author: think
"""
import matplotlib.pyplot as plt
import tensorly as tl
import numpy as np
from scipy.misc import face, imresize, imread, imsave
from tensorly.decomposition import parafac
from tensorly.decomposition import tucker
from tensorly.decomposition import weighted_non_negative_Matrix_parafac
from tensorly.base import unfold, fold
from math import ceil
from tqdm import tqdm
import time as time

#time cost- star
time_start=time.time()

res = 80 # define the same resolution with generated 4d lightfiled
def to_image(tensor):
    """A convenience function to convert from a float dtype back to uint8"""
    im = tl.to_numpy(tensor)
    im -= 0
    im /= im.max()
    im *= 255
    return im.astype(np.uint8)

def to_trans(Fdtensor):
    """A convenience function to convert from L255  to TRANSPARENTS"""
    trans_tensor = tl.to_numpy(Fdtensor)
    trans_tensor /= trans_tensor.max()
    return trans_tensor.astype(np.float16)


      
X = []
X = np.load('80x80_BOE 4dlightfield.npy')
print('################','\n','4d lightfiled load successfully!','\n\n','################')



# define a weighted tensor W binary 0 or 1 based on the diffrence bettween us and vt
dimention = res
weighted_W=np.zeros_like(X, dtype = 'float16')
for wu in tqdm(range(dimention),desc = 'weighted tensor W generating'):
    for wv in range(dimention):
        for ws in range(dimention):
            for wt in range(dimention):
                if wu-ws  < - dimention / 4 or wu-ws  > dimention  / 4:
                    weighted_W[wu, wv, ws, wt, :] = 0
                elif wv-wt  < - dimention / 4 or wv-wt  > dimention  / 4:
                    weighted_W[wu, wv, ws, wt, :] = 0
                else: 
                    weighted_W[wu, wv, ws, wt, :] = 1


np.save('weighted_W', weighted_W)
'''
weighted_W = np.load('weighted_W.npy')
'''
fig = plt.figure()
ax = fig.add_subplot(2, 3, 1)
ax.set_axis_off()
ax.imshow(to_image(weighted_W[0, 0, : , : , :]))
ax.set_title('weighted_W image00')

ax = fig.add_subplot(2, 3, 2)
ax.set_axis_off()
ax.imshow(to_image(weighted_W[0, :, 20 , : , :]))
ax.set_title('us weighted_W')

ax = fig.add_subplot(2, 3, 3)
ax.set_axis_off()
ax.imshow(to_image(weighted_W[26, :, 31 , : , :]))
ax.set_title('us weighted_W')


ax = fig.add_subplot(2, 3, 4)
ax.set_axis_off()
ax.imshow(to_image(weighted_W[12, :, : , 30 , :]))
ax.set_title('ut weighted_W')

ax = fig.add_subplot(2, 3, 5)
ax.set_axis_off()
ax.imshow(to_image(weighted_W[:, :, 23 , 30 , :]))
ax.set_title('st weighted_W')

ax = fig.add_subplot(2, 3, 6)
ax.set_axis_off()
ax.imshow(to_image(weighted_W[:, 23, : , 30 , :]))
ax.set_title('vs weighted_W')

plt.tight_layout()
plt.show()

print('################','\n','weighted tensor W generated successfully!','\n','################')
     
#X = np.array(X) * np.array(weighted_W)
 
X = to_trans(X) 


fig = plt.figure()
ax = fig.add_subplot(2, 3, 1)
ax.set_axis_off()
ax.imshow(to_image(X[0, 0, : , : , :]))
ax.set_title('11 image')

ax = fig.add_subplot(2, 3, 2)
ax.set_axis_off()
ax.imshow(to_image(X[0, :, 20 , : , :]))
ax.set_title('us')

ax = fig.add_subplot(2, 3, 3)
ax.set_axis_off()
ax.imshow(to_image(X[26, :, 31 , : , :]))
ax.set_title('us')


ax = fig.add_subplot(2, 3, 4)
ax.set_axis_off()
ax.imshow(to_image(X[12, :, : , 30 , :]))
ax.set_title('ut')

ax = fig.add_subplot(2, 3, 5)
ax.set_axis_off()
ax.imshow(to_image(X[:, :, 23 , 30 , :]))
ax.set_title('st')

ax = fig.add_subplot(2, 3, 6)
ax.set_axis_off()
ax.imshow(to_image(X[:, 23, : , 30 , :]))
ax.set_title('vs')

plt.tight_layout()
plt.show()


X = tl.tensor(X, dtype = 'float16') # change the dtype of X 
#RGB THREE CHANNEL CP Decomposition separately
R_weighted_W = weighted_W[:, :, :, :, 0]
G_weighted_W = weighted_W[:, :, :, :, 1]
B_weighted_W = weighted_W[:, :, :, :, 2]

R_X = X[:, :, :, :, 0]
G_X = X[:, :, :, :, 1]
B_X = X[:, :, :, :, 2]

# Rank of the CP decomposition
rank = 1
maxiternum = 300
iternum = 1
RUV = np.zeros([res, res])
RST = np.zeros([res, res])
GUV = np.zeros([res, res])
GST = np.zeros([res, res])
BUV = np.zeros([res, res])
BST = np.zeros([res, res])
Angle00 = []


      
for frame in range(iternum):
    random_state = 5 + frame
    # Perform the CP decomposition
    #factors = parafac(X, rank=cp_rank, init='random', tol=10e-3)
    #R  tensor, rank, n_iter_max=100, init='svd'/random, tol=10e-7, random_state=None, verbose=0
    #SVD method( donot use svd ) could make a raipied reduce variation tol e-5 is enought max iteration number 300
    #rank 16 could work , high is better but not meaningful
    
    RUV, RST = weighted_non_negative_Matrix_parafac(R_X, rank, R_weighted_W, n_iter_max = maxiternum, init = 'random',
                                      tol = 1e-5,random_state=None, verbose=1)

    #G
    GUV, GST = weighted_non_negative_Matrix_parafac(G_X,  rank, G_weighted_W, n_iter_max = maxiternum, init = 'random',
                                      tol = 1e-5,random_state=None, verbose=0)

    #B
    BUV, BST = weighted_non_negative_Matrix_parafac(B_X, rank, B_weighted_W, n_iter_max = maxiternum, init = 'random',
                                      tol = 1e-5,random_state=None, verbose=0)
    
    ###########save everyimage 
    Frame_RUV = RUV
    Frame_RST = RST
    Frame_GUV = GUV
    Frame_GST = GST
    Frame_BUV = BUV
    Frame_BST = BST
    Frame_UV =np.concatenate(( Frame_RUV[:, :, np.newaxis], Frame_GUV[:, :, np.newaxis], Frame_BUV[:, :, np.newaxis]),axis=2)
    Frame_ST =np.concatenate(( Frame_RST[:, :, np.newaxis], Frame_GST[:, :, np.newaxis], Frame_BST[:, :, np.newaxis]),axis=2)
    FrameUV = str(frame) + 'UV'
    FrameST = str(frame) + 'ST'
    imsave('.\RSLT\FrameUV{}.png'.format(frame), Frame_UV)
    imsave('.\RSLT\FrameST{}.png'.format(frame), Frame_ST)
    #####save everyimage 
    print('\n\n****************','\n')
    print(frame,'NTF Decomposition complite !!')
    print('****************','\n')
'''
RUV = np.array(RUV / iternum)
RST = np.array(RST / iternum)
GUV = np.array(GUV / iternum)
GST = np.array(GST / iternum)
BUV = np.array(BUV / iternum)
BST = np.array(BST / iternum)
'''
# unfold to UV and ST 
RGB_UV = []
RGB_ST  = []
RGB_UV = np.array(RGB_UV)
RGB_ST = np.array(RGB_ST)
RGB_UV =np.concatenate((RUV[:, :, np.newaxis],GUV[:, :, np.newaxis],BUV[:, :, np.newaxis]),axis=2)
RGB_ST =np.concatenate((RST[:, :, np.newaxis],GST[:, :, np.newaxis],BST[:, :, np.newaxis]),axis=2)


print (RGB_UV.shape)
print (RGB_ST.shape)
print ((RGB_UV + RGB_ST).shape)

'''
# Reconstruct the image from the factors
cp_reconstruction = tl.kruskal_to_tensor(nnfactors)
print(cp_reconstruction.shape)
'''
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax.set_axis_off()
ax.imshow(to_image(X[35, 35, : , : , :]))
ax.set_title('Orignal')
plt.tight_layout()

ax = fig.add_subplot(1, 2, 2)
ax.set_axis_off()
ax.imshow(to_image(RGB_UV  * RGB_ST))
ax.set_title('Recondtructed with Rank = rank')
plt.tight_layout()


plt.show()

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax.set_axis_off()
ax.imshow(to_image(RGB_UV))
ax.set_title('UV')
plt.tight_layout()

ax = fig.add_subplot(1, 2, 2)
ax.set_axis_off()
ax.imshow(to_image( RGB_ST))
ax.set_title('ST')
plt.tight_layout()


plt.show()
'''
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax.set_axis_off()
ax.imshow(to_image(X[0, 0, : , : , :]))
ax.set_title('uv')
plt.tight_layout()

ax = fig.add_subplot(1, 2, 2)
ax.set_axis_off()
ax.imshow(to_image(X[35, 35, : , : , :]))
ax.set_title('st')
plt.tight_layout()


plt.show()
'''

#time cost- End
time_end=time.time()
print('totally cost:',int((time_end-time_start)//60),'min',(time_end-time_start)%60,'s')
