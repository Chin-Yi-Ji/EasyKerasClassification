# *EasyKerasClassification*

An easier way to train Image Classification by using Tensorflow + Keras
It is a Repository for training Image Classification model easily.

#### 紀欽益 Chin-Yi-Ji

#### E-mail: allen82218@gmail.com

#### Version
- 2021 0622  ver1.0.0 : 上傳 


---

[Introduction](#Introduction)

[Requirment](#Requirment)

[Resource](#Resource)

[Training](#Training)

[Inference](#Inference)

[Other](#Other)

---

## Introduction

An easier way to train Image Classification by using Tensorflow + Keras
It is a Repository for training Image Classification model easily.

Backbone models are from tf.keras.applications.XXXXX, all the models here can be called here.
Optimizers are from tensorflow.keras.optimizers.XXXX and tensorflow_addons.optimizers.XXXX
Losses are from tensorflow.keraslosses.XXXXXX


## Requirment

check [requirements.txt](https://github.com/Chin-Yi-Ji/EasyKerasClassification/blob/main/requirement.yml)  

see requirement.yml
if you are using conda, you can use the following command to install the requiremnet

```
conda env create -f /path/to/requirement.yml
```



## Resource

All the training detail can be found in [Tensorflow](https://www.tensorflow.org/tutorials?hl=zh-tw)


## Training

The hardest part in this repository is to organize data into a specific format.
For example, you are going to create a animal identification model, and there are dog, cat, wolf and bear images to be recognized.
You have to set your folder like the following:

* Animal_Identify (or any name you like) 
  * dog
  * cat
  * wolf
  * bear

Next step, run folder_to_csv.py to get a csvfile
``` bash
python folder_to_csv.py -ffolder "Animal_Identify/" -sp "SAVEPATH"
```
The last step in 
## Inference

The codes you need to load the saved model is the following:

```python
import tensorflow as tf
import tensorflow_addons as tfa
model = tf.keras.models.load_model( YOURMODELPATH )

loss, categorical_accuracy = model.evaluate( dataset_with_label )
class_prob = model.predict( dataset_without_label )
```

## Other
The momentum will late update.
