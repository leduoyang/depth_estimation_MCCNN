import numpy as np
import argparse
import cv2
import cv2.ximgproc as cv2_x
import time
import json
from util import writePFM,readPFM
import pdb
from keras.models import model_from_json
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical,Sequence
from keras.layers.merge import concatenate
from matplotlib import pyplot as plt
from keras.models import load_model
from sklearn.feature_extraction import image
from tqdm import tqdm
from scipy import stats
parser = argparse.ArgumentParser(description='Disparity Estimation')
parser.add_argument('--input-left', default='./data/Synthetic/TL1.png', type=str, help='input left image')
parser.add_argument('--input-right', default='./data/Synthetic/TR1.png', type=str, help='input right image')
parser.add_argument('--output', default='./TL1.pfm', type=str, help='left disparity map')

# load model for cost computation
print('load model...') 
with open('model4cost.json', "r") as f:
    model = model_from_json(json.loads(f.read()));
model.load_weights('model4cost');

patch_size = 9 # according to the size from training data
feature_dim = 200
################################### define models for extracting features and predicting match cost from features pair ############################################
# feature model
INPUT1_1 = Input(shape = (9,9,1),name='INPUT1_1')
CONV1_1 = Conv2D(32,kernel_size=5,padding='valid',activation='relu',name='CONV1_1')(INPUT1_1)
FLAT1_1 = Flatten()(CONV1_1)
DEN1_2 = Dense(200,activation='relu',name='DEN1_2')(FLAT1_1)
DEN1_3 = Dense(200,activation='relu',name='DEN1_3')(DEN1_2)
model4left_feature = Model(INPUT1_1,DEN1_3)
model4left_feature.load_weights('model4cost',by_name=True)
    
INPUT2_1 = Input(shape = (9,9,1),name='INPUT2_1')
CONV2_1 = Conv2D(32,kernel_size=5,padding='valid',activation='relu',name='CONV2_1')(INPUT2_1)
FLAT2_1 = Flatten()(CONV2_1)    
DEN2_2 = Dense(200,activation='relu',name='DEN2_2')(FLAT2_1)
DEN2_3 = Dense(200,activation='relu',name='DEN2_3')(DEN2_2)
model4right_feature = Model(INPUT2_1,DEN2_3)
model4right_feature.load_weights('model4cost',by_name=True)

# prediction model
INPUT = Input(shape = (1,400))
DEN4 = Dense(300,activation='relu',name='DEN4')(INPUT)
DEN5 = Dense(300,activation='relu',name='DEN5')(DEN4)
DEN6 = Dense(300,activation='relu',name='DEN6')(DEN5)
DEN7 = Dense(300,activation='relu',name='DEN7')(DEN6)
FC8 = Dense(2,activation='softmax',name='FC8')(DEN7)
model4predict = Model(INPUT,FC8)  
model4predict.load_weights('model4cost',by_name=True)

################################### Extract left / right pixel features ############################################
def extract_feature(patches,flag):
    print('extract_feature...')
    m = model4left_feature
    if flag == 'r': #left image
        m = model4right_feature
    feature_vectors = np.zeros((patches.shape[0],feature_dim))
    feature_vectors = m.predict(patches.reshape((patches.shape[0],patches.shape[1],patches.shape[2],1)))
    return feature_vectors
    
################################### Cost computation ############################################
def cost_computation(fvl,fvr,param):
    print('cost_computation...') 
    h, w , max_disp= param
    l2r_cost = np.zeros((h,w,max_disp))
    r2l_cost = np.zeros((h,w,max_disp))

    for d in tqdm(range(max_disp)): 
        #piece-wise squared differences 
        left = fvl[:,d:,:]
        right = fvr[:,:w-d,:]
        # compute features for each patch with particular pixel
        flatten_left = left.reshape((left.shape[0]*left.shape[1],left.shape[2]))
        flatten_right = right.reshape((left.shape[0]*left.shape[1],left.shape[2]))
        
        tem = np.concatenate((flatten_left,flatten_right),axis=1)
        tem = model4predict.predict(tem.reshape((tem.shape[0],1,tem.shape[1])))[:,:,0]
        l2r_cost[:,d:,d] = tem.reshape((left.shape[0],left.shape[1]))
        r2l_cost[:,0:w-d,d] = tem.reshape((left.shape[0],left.shape[1]))
        if d > 0 : #padding
            l2r_cost[:,0:d,d] = l2r_cost[:,d,d].reshape(-1,1)
            r2l_cost[:,w-d:w,d] = r2l_cost[:,w-1-d,d].reshape(-1,1)
    return l2r_cost , r2l_cost 

################################### Cost aggregation ############################################
def cost_volumne_filtering(raw_cost):
    print('cost_volumne_filtering...') 
    h,w,max_disp = raw_cost.shape
    smoothed_cost = np.zeros((h,w,max_disp))     
    for d in range(max_disp):
        smoothed_cost[:,:,d] = cv2.blur(raw_cost[:,:,d],(5,5))
    
    return smoothed_cost

###################################Disparity optimization############################################
def WTA(l2r_cost,r2l_cost):
    print('Winner Take All...') 
    h , w , dp =l2r_cost.shape
    l2r_labels = np.zeros((h,w))
    r2l_labels = np.zeros((h,w))    
    # Winner-take-all.
    for i in range(h):
        for j in range(w):
            cost = l2r_cost[i,j,:]
            l2r_labels[i,j]= np.argmin(cost)
            cost = r2l_cost[i,j,:]
            r2l_labels[i,j]= np.argmin(cost)
    return l2r_labels , r2l_labels

###################################Disparity refinement############################################
def consistency_check(Dl,Dr):
    h,w = Dl.shape
    Dl = np.float64(Dl)
    Y , X = [] , []
    for y in range(h):
        for x in range(w):
            if x-int(Dl[y,x]) >= 0:
                if  Dl[y,x] != Dr[y,x-int(Dl[y,x])]:
                    Dl[y,x] = 0
                    X.append(x)
                    Y.append(y)
    return  Dl,Y,X

def hole_filling(labels,Y,X):
    L = len(Y)
    for l in range(L):
        y , x = Y[l] , X[l]
        l_slice,r_slice = labels[y,0:x],labels[y,x+1:labels.shape[1]]
        l_can , r_can = -1, -1
        i , j = len(l_slice)-1 , 0
        while(l_can <= 0 and i >= 0 ):
            l_can = l_slice[i]
            i = i - 1
        while(r_can <= 0 and j < len(r_slice)):
            r_can = r_slice[j]
            j = j + 1            
        if l_can <=0 :
            labels[y,x] = r_can
        elif r_can <=0 :
            labels[y,x] = l_can
        else:        
            labels[y,x] = min(l_can,r_can)
    return labels

def exeSegementation(img,sigma,k,min_size):
    segmentator = cv2_x.segmentation.createGraphSegmentation(sigma, k, min_size)
    segment = segmentator.processImage(img)
    return segment

def refine_with_seg_img(labels,seg_img,kernel_size):
    h , w = labels.shape
    labels_after_refinement = np.zeros((h,w))
    aug_labels = cv2.copyMakeBorder(labels,kernel_size//2,kernel_size//2,kernel_size//2,kernel_size//2,cv2.BORDER_REPLICATE)
    aug_seg_img = cv2.copyMakeBorder(seg_img,kernel_size//2,kernel_size//2,kernel_size//2,kernel_size//2,cv2.BORDER_REPLICATE)    
    for i in range(h):
        for j in range(w):
            seg_win , label_win = aug_seg_img[i:i+kernel_size,j:j+kernel_size] , aug_labels[i:i+kernel_size,j:j+kernel_size]  
            indx_x , indx_y = np.where(seg_win == seg_win[kernel_size//2,kernel_size//2])
            labels_after_refinement[i,j] = 0.5*seg_win[kernel_size//2,kernel_size//2]+0.5*stats.mode(label_win[indx_x,indx_y])[0]
    return labels_after_refinement

# You can modify the function interface as you like
def computeDisp(Il, Ir):
    sigma , k , min_size = 0.5 , 100 , 200
    seg_img = exeSegementation(Il,sigma,k,min_size)

    print('computeDisp...')    
    h, w = Il.shape
    ratio = (62 / 384) # max disp. in ground truth / width of the corresponding image
    max_disp = int(w * ratio)
    disp = np.zeros((h, w), dtype=np.float32)
    
    # >>> normalization
    nor_Il , nor_Ir = Il / 255 , Ir / 255
    nor_Il = (nor_Il - np.mean(nor_Il)) / np.std(nor_Il)
    nor_Ir = (nor_Ir - np.mean(nor_Ir)) / np.std(nor_Ir)

    # compute features for each patch with particular pixel
    aug_Il = cv2.copyMakeBorder(nor_Il,patch_size//2,patch_size//2,patch_size//2,patch_size//2,cv2.BORDER_CONSTANT,value=0)
    aug_Ir = cv2.copyMakeBorder(nor_Ir,patch_size//2,patch_size//2,patch_size//2,patch_size//2,cv2.BORDER_CONSTANT,value=0)
    left_image_patches = image.extract_patches_2d(aug_Il, (patch_size, patch_size))
    right_image_patches = image.extract_patches_2d(aug_Ir, (patch_size, patch_size))
    
    left_feature_vec = extract_feature(left_image_patches,'l').reshape((h,w,feature_dim))
    right_feature_vec = extract_feature(right_image_patches,'r').reshape((h,w,feature_dim))
    
    # >>> Cost computation
    l2r_cost , r2l_cost = cost_computation(left_feature_vec,right_feature_vec,(h,w,max_disp)) 
    
    # >>> Cost aggregation
    l2r_cost_aggregation = cost_volumne_filtering(l2r_cost)
    r2l_cost_aggregation = cost_volumne_filtering(r2l_cost)
    
    # >>> Disparity optimization
    l2r_disp , r2l_disp = WTA(l2r_cost_aggregation , r2l_cost_aggregation)# Winner-take-all.

    # >>> Disparity refinement
    l_disp = cv2_x.weightedMedianFilter(Il.astype('uint8'),l2r_disp.astype('uint8'),15,5,cv2_x.WMF_JAC)
    r_disp = cv2_x.weightedMedianFilter(Ir.astype('uint8'),r2l_disp.astype('uint8'),15,5,cv2_x.WMF_JAC)
    labels,Y,X = consistency_check(l_disp,r_disp)# Left-right consistency check 
    labels = hole_filling(labels,Y,X)
    labels = cv2_x.weightedMedianFilter(Il.astype('uint8'),labels.astype('uint8'),15,5,cv2_x.WMF_JAC)

    kernel_size = 11
    labelsss = refine_with_seg_img(labels,seg_img,kernel_size)
    pdb.set_trace()
    return labels


def main():
    args = parser.parse_args()

    print('Compute disparity for %s' % args.input_left)
    # read images
    img_left = cv2.imread(args.input_left,cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img_right = cv2.imread(args.input_right,cv2.IMREAD_GRAYSCALE).astype(np.float32)
        
    # start computing disparity map
    tic = time.time()
    disp = computeDisp(img_left, img_right)
    toc = time.time()
    """
    plt.figure(0)
    plt.imshow(disp,cmap='gray')
    plt.show()"""
    
    writePFM(args.output, disp.astype('float32'))
    print('Elapsed time: %f sec.' % (toc - tic))
    


if __name__ == '__main__':
    main()
