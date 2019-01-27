"""
Train a model for cost computation
usage:
    python train.py "path of the dataset"
e.g.
    python train.py ./data/synthetic
"""
import json
import sys
import numpy as np
from datagenerator import ImageDataGenerator
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical,Sequence
from keras.layers.merge import concatenate
from matplotlib import pyplot as plt
import pdb
class generator4model(Sequence):
    def __init__(self, datagen, batch_size,interval):
        self.datagen = datagen
        self.batch_size = batch_size
        self.n_classes = 2
        self.interval = interval
    def __len__(self): #Denotes the number of batches per epoch 
        return 1000
    def __getitem__(self,idx): # return a complete batch
        left_patchs , right_neg_patchs, right_pos_patchs = self.datagen.extract_batchs(self.batch_size,self.interval)
        X1 = np.concatenate(( left_patchs,left_patchs ),axis=0)
        X2 = np.concatenate(( right_neg_patchs,right_pos_patchs ),axis=0)
        y = [0] * len(left_patchs) + [1] * len(left_patchs)
        return [X1,X2],to_categorical(y, num_classes=self.n_classes)

def buildModel():
    INPUT1_1 = Input(shape = (9,9,1),name='INPUT1_1')
    CONV1_1 = Conv2D(32,kernel_size=5,padding='valid',activation='relu',name='CONV1_1')(INPUT1_1)
    FLAT1_1 = Flatten()(CONV1_1)
    DEN1_2 = Dense(200,activation='relu',name='DEN1_2')(FLAT1_1)
    DEN1_3 = Dense(200,activation='relu',name='DEN1_3')(DEN1_2)
    
    INPUT2_1 = Input(shape = (9,9,1),name='INPUT2_1')
    CONV2_1 = Conv2D(32,kernel_size=5,padding='valid',activation='relu',name='CONV2_1')(INPUT2_1)
    FLAT2_1 = Flatten()(CONV2_1)    
    DEN2_2 = Dense(200,activation='relu',name='DEN2_2')(FLAT2_1)
    DEN2_3 = Dense(200,activation='relu',name='DEN2_3')(DEN2_2)
    
    CONCAT = concatenate([DEN1_3, DEN2_3])
    DEN4 = Dense(300,activation='relu',name='DEN4')(CONCAT)
    DEN5 = Dense(300,activation='relu',name='DEN5')(DEN4)
    DEN6 = Dense(300,activation='relu',name='DEN6')(DEN5)
    DEN7 = Dense(300,activation='relu',name='DEN7')(DEN6)
    FC8 = Dense(2,activation='softmax',name='FC8')(DEN7)
    
    model = Model(inputs=[INPUT1_1, INPUT2_1],outputs = FC8)


    return model
    
if __name__ == '__main__':    
    # import training data
    argv = sys.argv
    if len(argv) < 1:
        raise Exception('please enter directory of dataset respectively')
    tf = argv[1]
    if tf[-1] != '/':
        tf = tf + '/'
    
    dg = ImageDataGenerator(tf,shuffle=True)
    
    # build model
    model  = buildModel()
    model.summary()    
    
    model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=1e-4),metrics=['accuracy'])       
    ckpt = ModelCheckpoint('model4cost',monitor='val_acc',save_best_only=True,save_weights_only=True,verbose=1)
    cb= [ckpt]
    
    # save model architecture
    json_string = model.to_json()
    with open('model4cost.json', 'w') as outfile: 
        json.dump(json_string, outfile)    
    model.save('model.h5')

    
    # start training
    batch_size = 128
    num_epochs = 20
    gen1 = generator4model(dg,batch_size,(0,7))
    gen2 = generator4model(dg,batch_size,(7,10))    
    history = model.fit_generator(gen1,epochs=num_epochs,validation_data=gen2,callbacks=cb,verbose=1)    
    
    # evaluate model with validation set
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    
    plt.figure(figsize=(12,5))
    plt.subplot(121)
    plt.plot(np.arange(num_epochs),loss,'b',label='train loss')
    plt.plot(np.arange(num_epochs),val_loss,'r',label='valid loss')    
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("loss curve")
    plt.legend(loc='best')
    
    plt.subplot(122)
    plt.plot(np.arange(num_epochs)+1,acc,'b',label='train accuracy')
    plt.plot(np.arange(num_epochs)+1,val_acc,'r',label='valid accuracy')    
    plt.xlabel("epochs")
    plt.ylabel("accuarcy")
    plt.title("accuarcy curve")  
    plt.legend(loc='best')
    plt.tight_layout()
    
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    