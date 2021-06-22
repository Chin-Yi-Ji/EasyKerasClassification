#!/usr/bin/env python
# coding: utf-8

# In[28]:


import tensorflow as tf
import tensorflow_addons as tfa
import pandas as pd
import numpy as np
import glob
import argparse
from modelzoo import import_model
from utils import Optimizer, Loss 
import time
from generator import MakeInferSet, MakeDataSet
import os

# In[5]:


parser =  argparse.ArgumentParser()
#Basic
parser.add_argument('-name','--name', type =str ,default = None, help = 'What kind of objects are being classified' )
parser.add_argument('-m','--model', type =str ,default = 'Model', help = 'Model' )
#parser.add_argument("-g", "--generator", type=str, default = "Generator", help = "Generator")
parser.add_argument('-num_class', '--num_class', type= int, default = 1, help = "Number of class")
parser.add_argument("-pretrain", "--pretrained", type = str, default = None, help = "Pretrained")
parser.add_argument("-loss", "--loss", type = str, default = None, help = "Use what kind of loss")
parser.add_argument("-pooling", "--pooling", type = str, default = None, help = "Pooling at the end of model")
parser.add_argument("-gpus", "--gpus", type = int,nargs = '+', default = None, help = "Use which GPU")

#Folder
parser.add_argument('-Trainlist','--Trainlist', type = str, default = None, help = 'Trainset "list" path')
parser.add_argument('-Testlist','--Testlist', type = str, default = None, help = 'Testset "list" path')
parser.add_argument('-Validlist','--Validlist', type = str, default = None, help = 'Validset "list" path')
#parser.add_argument("-Train", "--Train", type=int, default = 1, help = "Train or not,1 for train, 0 for not to train" )

#Hyperparameter
parser.add_argument('-resize','--resize', type = int , default = 512, help = 'Resize')
parser.add_argument('-epoch','--epoch',type = int, default = 30 , help = 'EPOCHS')
parser.add_argument('-b','--batch',type = int, default = None , help = 'Batch')
parser.add_argument('-lr','--learningrate',type = float, default = 1e-4, help = 'Learning_Rate')
parser.add_argument('-O','--optimizer', type = str, default = 'Adam')

#Output option
parser.add_argument("-N", "--Note", type=str, default = None, help = "Note")
parser.add_argument("-D", "--OutPutDir", type=str, default = None ,help = "the toppest OutPutDir")

args = parser.parse_args()


# In[ ]:


#GPU config
gpu_use = []
physical_devices = tf.config.list_physical_devices(device_type = 'GPU')

for num in args.gpus:
    gpu_use.append(physical_devices[num])    
tf.config.experimental.set_visible_devices(devices = gpu_use, device_type='GPU')



for i in physical_devices:
    tf.config.experimental.set_memory_growth(i, True)

#分散學習
mirrored_strategy = tf.distribute.MirroredStrategy()                 


# In[ ]:


train_csv = pd.read_csv(args.Trainlist)
valid_csv = pd.read_csv(args.Validlist)

train_img = train_csv['F_path']
train_mask = train_csv['Tcode']
train_mask = tf.one_hot(train_mask,depth = args.num_class)
#train_mask = tf.reshape(train_mask, [-1,1,args.num_class])
print("Train Mask Shape: ",train_mask.shape)

val_img = valid_csv['F_path']
val_mask = valid_csv['Tcode']
val_mask = tf.one_hot(val_mask, depth = args.num_class)
#val_mask = tf.reshape(val_mask, [-1,1,args.num_class])
# dataset
train_set = MakeDataSet(train_img, train_mask, SIZE = args.resize, BATCH = args.batch,AUGMENT=True,SHUFFLE=True)
print('Train:',len(train_img),len(train_mask))

# #valid set
valid_set = MakeDataSet(val_img, val_mask, SIZE = args.resize, BATCH = args.batch,AUGMENT=False,SHUFFLE=False)
print('Valid:',len(val_img),len(val_mask))


# In[ ]:


#處理directory
now = time.strftime('%Y%m%d')
SaveName = args.name + "_" + args.model + "_" +"S" + str(args.resize) + "C" +str(args.num_class) + "E" + str(args.epoch)+ "O" + args.optimizer +"LR" + str(args.learningrate) + "_" + now + "/"

path = args.OutPutDir
#model_directory

model_dir = path + SaveName + "/Best_model"
metric_dir = path + SaveName

os.makedirs(path, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(metric_dir, exist_ok=True)
os.makedirs(metric_dir + "/figure", exist_ok=True)
os.makedirs(metric_dir + "/figure/performance", exist_ok=True)
os.makedirs(metric_dir + "/figure/prediction", exist_ok=True)


# In[ ]:


log_dir = os.path.join(path,SaveName,'TensorBoard_logs')

model_cbk = tf.keras.callbacks.TensorBoard( log_dir = log_dir )
model_mckp = tf.keras.callbacks.ModelCheckpoint(filepath = model_dir,
                                                monitor = "val_categorical_accuracy",
                                                mode = "max", 
                                                save_best_only = True,                                              
                                                save_weights_only = False)


# In[2]:


#strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
#with strategy.scope():
    #net = tf.keras.applications.ResNet50(include_top=False,weights='imagenet')
    
base_net = import_model(model = args.model, pretrain = args.pretrained, pool = args.pooling)

out = tf.keras.layers.Dense(args.num_class, activation = 'softmax')(base_net.output)

model = tf.keras.models.Model(inputs = base_net.input, outputs = out)

print("Model_Output_Shape:", model.output_shape)
#Optimizer
#for expample: opt = Optimizer(Ranger,1e-4)

opt = Optimizer(args.optimizer, args.learningrate)
# Loss
#for example: loss=Loss('CategoricalCrossentropy')
loss = Loss(args.loss)

model.compile(optimizer = opt,
              loss=loss,
              metrics=[tf.keras.metrics.CategoricalAccuracy()])



trained_model = model.fit(train_set,
                          validation_data=valid_set,
                          epochs=args.epoch,
                          callbacks=[model_cbk, model_mckp])



# In[ ]:


"""plot的部份應該要另外寫"""

train_loss = trained_model.history['loss']
train_accuracy = trained_model.history['categorical_accuracy']
val_loss = trained_model.history['val_loss']
val_accuracy = trained_model.history['val_categorical_accuracy']


# In[10]:


import matplotlib.pyplot as plt
plt.figure()
plt.plot(range(args.epoch),train_loss)
plt.plot(range(args.epoch),val_loss,color='r')
plt.xlim([0,args.epoch])
plt.xlabel('Epochs')
plt.ylabel('Loss')
ticks= list(range(60))
plt.xticks(ticks[0:(args.epoch+10):10])
plt.legend(['Train','Valid'])
plt.savefig(metric_dir + "/figure/performance/loss_ranger.png",dpi=300)
plt.close()


# In[11]:


plt.figure()
plt.plot(range(args.epoch),train_accuracy)
plt.plot(range(args.epoch),val_accuracy,color='r')
plt.xlim([0,args.epoch])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
ticks= list(range(60))
plt.xticks(ticks[0:(args.epoch+10):10])
plt.legend(['Train','Valid'])
plt.savefig(metric_dir + "/figure/performance/accuracy_ranger.png",dpi=300)


# In[ ]:


test_csv = pd.read_csv(args.Testlist)
test_img = test_csv['F_path']
test_mask = test_csv['Tcode']
test_mask = tf.one_hot(test_mask,depth = args.num_class)

test_set = MakeDataSet(test_img, test_mask, SIZE = args.resize, BATCH = args.batch , AUGMENT = False,SHUFFLE = False)

test_loss, test_accuracy = model.evaluate(test_set)


# In[ ]:


with open('{}/train_results.pkl'.format(metric_dir),'w') as f:
    pickle.dump(args,[train_loss,val_loss,test_loss,train_accuracy,val_accuracy,test_accuracy],f)

