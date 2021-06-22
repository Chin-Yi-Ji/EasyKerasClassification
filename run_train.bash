#!/bin/bash
path = /home/ortho_class_model/
cd $path
# Activate the training environment
# You should run "conda info --base" in shell to get the following path
source /root/anaconda3/etc/profile.d/conda.sh
# Activate your env for training
conda activate tensor_v2




for 
#
python train_model.py -name ortho_class \ #Naming what you are going to train
-m EfficientNetB6 \ # The keras model you are going to use
-num_class 17 \ # how many types of images
-pretrain imagenet \ # only imagenet and None could be called in keras
-loss CategoricalCrossentropy \
-pooling avg \
-Trainlist /home/ortho_class_model/T8V1TE1/train.csv \ #Use "saparate_dataset.py" for acquiring Trainlist,Testlist and Validlist
-Testlist /home/ortho_class_model/T8V1TE1/test.csv \
-Validlist /home/ortho_class_model/T8V1TE1/valid.csv \
-resize 224 \
-epoch 50 \
-b 16 \ #batch size
-lr 0.0001 \
-O Ranger \
-D /home/ortho_class_model/results/ \ # the Toppest output folder
