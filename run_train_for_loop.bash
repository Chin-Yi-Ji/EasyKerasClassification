#!/bin/bash
path = /home/ortho_class_model/
# Activate the training environment
# You should run "conda info --base" in shell to get the following path
cd $path
source /root/anaconda3/etc/profile.d/conda.sh
# Activate your env for training
conda activate tensor_v2

# #Make a gengeral list including path and type
# python folder_to_csv.py -ffolder /home/ortho_class_model/Notfrom_sdcard/ \
# -sp /home/ortho_class_model/dataset/


# # Separating images into Train, Valid and Test, export ".csv" file respectively
# python saparate_dataset.py -tp /home/ortho_class_model/total_data_ind.csv \
# -sp /home/otrho_class_model/dataset/ \
# -T 8 \ #the propotion of data assigned to Trainset
# -V 1 \ #the propotion of data assigned to Validset
# -Te 1  #the propotion of data assigned to TEstset




for LR in 0.0001 0.001 0.0003 0.0005
do \
nohup python /home/ortho_class_model/train_model.py -name ortho_class \
-gpus 0 1 2 \
-m ResNet50 \
-num_class 17 \
-pretrain imagenet \
-loss CategoricalCrossentropy \
-pooling avg \
-Trainlist /home/ortho_class_model/T8V1TE1/train.csv \
-Testlist /home/ortho_class_model/T8V1TE1/test.csv \
-Validlist /home/ortho_class_model/T8V1TE1/valid.csv \
-resize 224 \
-epoch 30 \
-b 8 \
-lr $LR \
-O Ranger \
-D /home/ortho_class_model/results/  >> /home/ortho_class_model/results/"$LR"_train_logs.out&
done
