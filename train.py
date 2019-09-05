import cv2
import tensorflow as tf
import keras
import pandas as pd
import numpy as np
from time import time
from model import ResNet152
path_csv='/home/shivam/Documents/MURA/MURA-v1.1/wrist.csv'
data=pd.read_csv(path_csv, usecols=['image', 'label'])

def flip(img):
    img=cv2.flip(img,0)
    return img


def preprocess(line_data):
    label=line_data['label'][0]
    img_path=line_data['image'][0]
    img=cv2.imread(img_path)
    k=np.random.randint(0,2)
    if(k==0):
        img=flip(img)
    #img=img.astype('float32')
    img = cv2.resize(img,(224,224))
    # img=img/255.0
    # img[:,:,0]=(img[:,:,0]-0.485)/0.229
    # img[:,:,1]=(img[:,:,1]-0.456)/0.224
    # img[:,:,2]=(img[:,:,2]-0.406)/0.225
    return img, label

def preprocess_test(line_data):
    label=line_data['label'][0]
    img_path=line_data['image'][0]
    img=cv2.imread(img_path)
    #img=img.astype('float32')
    img = cv2.resize(img,(224,224))
    # img=img/255.0
    # img[:,:,0]=(img[:,:,0]-0.485)/0.229
    # img[:,:,1]=(img[:,:,1]-0.456)/0.224
    # img[:,:,2]=(img[:,:,2]-0.406)/0.225
    return img, label


def train_gen(data,batch_size):
    batch_images=np.zeros((batch_size,224,224,3))
    batch_labels=np.zeros(batch_size)
    while True: 
        for i in range(0,batch_size):
            i_line=np.random.randint(len(data))
            line_data= data.iloc[[i_line]].reset_index()
            x,y=preprocess(line_data)
            batch_images[i]=x
            batch_labels[i]=y
        yield batch_images,batch_labels
def test_gen(data):
    while True:
        for i in range(0,len(data)):
            i_line=np.random.randint(len(data))
            line_data= data.iloc[[i_line]].reset_index()
            x,y=preprocess_test(line_data)
            x=x.reshape(1,x.shape[0], x.shape[1],x.shape[2])
            y=np.array([y])
            yield x,y
def custom_loss(w1,w2):
    def loss(y_true, y_pred):
        ans = -1*(w1*y_true*tf.log(y_pred) + w2*(1-y_true)*tf.log(y_pred))
        return ans
    return loss
        

def training():
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs/{}".format(time()))
    fun=custom_loss(w1=0.4,w2=0.6)
    model=ResNet152(include_top=False)
    model.summary()
    sgd=keras.optimizers.SGD(decay=1e-6, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd,metrics=['accuracy'])
    batch_size=8
    traingen=train_gen(data, batch_size)
    val_gen=test_gen(data)
    for k in range(10):
        model.fit_generator(traingen, steps_per_epoch=1300, epochs=5,validation_data=val_gen,validation_steps=256, callbacks=[tensorboard])
        weights_path='model_'+str(k)+'.h5'
        model.save_weights(weights_path)
                    
training()


            

    
