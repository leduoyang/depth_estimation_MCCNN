"""
Process Synthetic data into patch pairs for training
1. normalize image to 0 mean , 1 standard deviation
2. set max_disp used for testing
3. generate input sequence for training
"""
import os
import numpy as np
import cv2
from util import readPFM
from sklearn.utils import shuffle

class ImageDataGenerator:
    def __init__(self, filename, shuffle=False, patch_size=9, neg_low=4, neg_high=8, pos=1):
        self.fn = filename
        self.shuffle = shuffle
        self.patch_size = patch_size
        self.neg_low = neg_low
        self.neg_high = neg_high
        self.pos = pos
        
        self.pointer = 0
        self.read_filepath()
        if self.shuffle:
            self.exe_shuffle()
        self.fetch_data()
        print("init done...")
        
    def read_filepath(self):
        self.Ils_path, self.Irs_path, self.Gts_path = [], [], []
        for file in sorted(os.listdir(self.fn)):
            flag = file.split('.')[0][-2]
            if flag == 'L':
                num = file.split('.')[0][-1]
                # save file paths
                self.Ils_path.append(self.fn+file)
                self.Irs_path.append(self.fn+"TR" + num + ".png")
                self.Gts_path.append(self.fn+"TLD" + num + ".pfm")
        self.data_size = len(self.Ils_path)
        print("readfilepath done...")
        
    def exe_shuffle(self):
        self.Ils_path, self.Irs_path, self.Gts_path = shuffle(self.Ils_path, self.Irs_path, self.Gts_path, random_state=0)
        print("exe_shuffle done...")
        
    def fetch_data(self):
        self.left_images, self.right_images, self.ground_truths = [] , [] , []
        for n in range(self.data_size):
            left_image = cv2.imread(self.Ils_path[n],cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
            right_image = cv2.imread(self.Irs_path[n],cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
            self.ground_truths.append(readPFM(self.Gts_path[n]))
            
            #normalization
            left_image = (left_image - np.mean(left_image)) / np.std(left_image)
            right_image = (right_image - np.mean(right_image)) / np.std(right_image)            
            
            self.left_images.append(left_image)
            self.right_images.append(right_image)
            
        print("fetchdata done...")
            
    def extract_batchs(self,batch_size,interval=(0,10)):
        if self.pointer >= interval[1]:
            self.pointer = interval[0]
        left_image, right_image, ground_truth  =  self.left_images[self.pointer], self.right_images[self.pointer],self.ground_truths[self.pointer]
        h, w = left_image.shape
        
        row = np.random.permutation(h-1)[0:batch_size]
        col = np.random.permutation(w-1)[0:batch_size]
        for n in range(batch_size):
            while(ground_truth[row[n],col[n]] == float('inf') or (col[n] - ground_truth[row[n],col[n]]) < 0 ):
                row[n] = np.random.randint(0 , high=h)
                col[n] = np.random.randint(0 , high=w)
        
        # padding
        aug_left_image = cv2.copyMakeBorder(left_image,self.patch_size//2,self.patch_size//2,self.patch_size//2,self.patch_size//2,cv2.BORDER_CONSTANT,value=0)
        aug_right_image = cv2.copyMakeBorder(right_image,self.patch_size//2,self.patch_size//2,self.patch_size//2,self.patch_size//2,cv2.BORDER_CONSTANT,value=0)
        
        # pick patch pairs
        left_patchs = np.zeros([batch_size,self.patch_size,self.patch_size,1], dtype=np.float32)
        right_neg_patchs = np.zeros([batch_size,self.patch_size,self.patch_size,1], dtype=np.float32)
        right_pos_patchs = np.zeros([batch_size,self.patch_size,self.patch_size,1], dtype=np.float32)
        for n in range(batch_size):
            left_patchs[n,:,:,0] = aug_left_image[row[n]:row[n]+self.patch_size,col[n]:col[n]+self.patch_size]
            gt = ground_truth[row[n],col[n]]
            
            #pick neg patch
            neg_col = -1
            while(neg_col<0 or neg_col>= w):
                shift = np.random.randint(self.neg_low,self.neg_high)
                if np.random.randint(-1,1) == -1:
                    shift = -1 *shift
                neg_col = int(col[n] - gt + shift)
            right_neg_patchs[n,:,:,0] = aug_right_image[row[n]:row[n]+self.patch_size,neg_col:neg_col+self.patch_size]
            
            #pick pos patch
            pos_col = -1
            while(pos_col<0 or pos_col>= w):
                shift = np.random.randint(-1*self.pos,self.pos) 
                pos_col = int(col[n] - gt + shift)
            right_pos_patchs[n,:,:,0] = aug_right_image[row[n]:row[n]+self.patch_size,pos_col:pos_col+self.patch_size]
        self.pointer = self.pointer + 1    
        return left_patchs , right_neg_patchs, right_pos_patchs
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            