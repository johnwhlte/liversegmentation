import numpy as np
import cv2
from glob import glob
import random
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Input, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta
from tensorflow.keras import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils  import to_categorical
from tensorflow.keras.metrics import AUC, Recall, Precision, BinaryAccuracy
#from sklearn.model_selection import train_test_split
import os
import tensorflow as tf
import collections

model = tf.keras.models.load_model('./saved_model/my_model')

class DataGen(tf.keras.utils.Sequence):
    def __init__(self, ids, path, batch_size=8, image_size=256):
        self.ids = ids
        self.path = path
        self.batch_size = batch_size
        self.image_size = image_size
        self.on_epoch_end()
        
    def __load__(self, id_name):
        ## Path
        image_path = os.path.join(self.path, id_name)
        
        ## Reading Image
        image = cv2.imread(image_path, 1)
        #image = cv2.resize(image, (self.image_size, self.image_size))
        
        _, w, _ = np.shape(image)
        
        w = int(w)
        #mask = image[:, w:, :]
        im = image[:, :w, :]

        #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        ## Normalizaing 
        im = im/255.0
        #mask = mask/255.0
        mask = 0
        
        return im, mask
    
    def __getitem__(self, index):
        if(index+1)*self.batch_size > len(self.ids):
            self.batch_size = len(self.ids) - index*self.batch_size
        
        files_batch = self.ids[index*self.batch_size : (index+1)*self.batch_size]
        
        image = []
        mask  = []
        
        for id_name in files_batch:
            _img, _mask = self.__load__(id_name)
            image.append(_img)
            mask.append(_mask)
            
        image = np.array(image)
        mask  = np.array(mask)
        
        return image, mask
    
    def on_epoch_end(self):
        pass
    
    def __len__(self):
        return int(np.ceil(len(self.ids)/float(self.batch_size)))

test_path = f'./static/c0_vips_files/17/'
test_ids = os.listdir(test_path)
image_size = 256
batch_size = 1
test_gen = DataGen(test_ids, test_path, image_size=image_size, batch_size=batch_size)

for i, image in enumerate(test_ids):
    x, y= test_gen.__getitem__(i)
    #print(np.shape(x))

    if np.shape(x) == (1, 256, 256, 3):
        predictions = model.predict(x)
        orig_img = x[0]*255
        seg_map = predictions[0]*255
        seg_map = np.array(seg_map, dtype='uint8')
        seg_map = cv2.cvtColor(seg_map, cv2.COLOR_GRAY2BGR)
        # cv2.imshow('',seg_map)
        # cv2.waitKey()

        # white_dict = collections.defaultdict(int)
        # blue_dict = collections.defaultdict(int)

        # for i, row in enumerate(orig_img):
        #     for j, val in enumerate(row):
        #         if val[0] >= 235 and val[1] >= 235 and val[2] >= 235:
        #             seg_map[i][j] = np.array([255, 255, 255])
        #         elif seg_map[i][j][0] >= 235 and seg_map[i][j][1] >= 235 and seg_map[i][j][2] >= 235:
        #             seg_map[i][j] = np.array([113, 188, 212])

        #im = np.array(seg_map, dtype='uint8')
        #im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        #gt = cv2.cvtColor(np.array(y[0, :, :]*255, dtype='uint8'), cv2.COLOR_GRAY2BGR)
        #inp = np.array(x[0, :, :]*255, dtype='uint8')
        #im = cv2.hconcat((gt, im))
        #im = cv2.hconcat((inp, im))
        im = cv2.cvtColor(seg_map, cv2.COLOR_BGR2RGB)
        image_name = image.split('.')[0]
        image_num = image_name.split('_')[1]
        if int(image_num) == 50:
            print(image_name) 
        cv2.imwrite(f'./static/infc0_files/17/{image_name}.jpeg', im)
        #plt.xlabel('Stained Image                 Weak Labels                   U-net CNN Pred')

        #plt.imshow(im)