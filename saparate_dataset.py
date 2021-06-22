#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as mp
import argparse
import os


# In[ ]:


parser =  argparse.ArgumentParser()
#
parser.add_argument('-tp','--targetpath', type =str ,default = None, help = 'where the total csv is' )
parser.add_argument('-sp','--savepath', type =str ,default = None, help = 'where the new csv to be saved' )

parser.add_argument('-T','--Train', type =int ,default = None, help = 'Portion in train' )
parser.add_argument('-V','--Valid', type =int ,default = None, help = 'Portion in valid' )
parser.add_argument('-Te','--Test', type =int ,default = None, help = 'Portion in test' )
args = parser.parse_args()


# In[ ]:


def data_split(dfp, train, valid ,test):
    df = pd.read_csv(dfp)
    df = df.sample(frac=1).reset_index(drop=True)
    total = len(df.index)
    deno = train + valid + test
    train_end = int((train/deno)*total)
    valid_end = int((valid/deno)*total) + train_end
    
    train_set = df.iloc[0:train_end]
    
    valid_set = df.iloc[train_end:valid_end]
    
    test_set = df.iloc[valid_end:]
    
    
    
    return train_set, valid_set, test_set


# In[ ]:


if __name__=='__main__':
    print('train:',args.Train,'valid:', args.Valid, 'test:',args.Test)
    save_folder = args.savepath + 'T' +str(args.Train) + 'V' + str(args.Valid) + 'TE' + str(args.Test)
    print(save_folder)
    train_set ,valid_set , test_set = data_split( args.targetpath, args.Train, args.Valid, args.Test )
    os.makedirs(save_folder ,exist_ok =False)
    train_set.to_csv(save_folder + '/train.csv', index=False)
    valid_set.to_csv(save_folder + '/valid.csv', index=False)
    test_set.to_csv(save_folder + '/test.csv', index=False)
    print('done')

