#!/usr/bin/env python
# coding: utf-8

import glob
import pandas as pd
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-ffolder','--filefolder',type = str, default = None, help = 'The data folder, the next layer is  what we storage images by type in each folder')
parser.add_argument('-sp','--savepath',type = str, default = None , help = 'The processed csv is going to be saved,end by a folder')
args = parser.parse_args()
print(args)
#抓出dataset分成各type的資料夾
file = glob.glob(args.filefolder + '*')
file = sorted(file)
file_info = pd.DataFrame()
for t in file:
    f_path = glob.glob(t + '/*.JPG')
    file_info = file_info.append(pd.DataFrame({'F_path':f_path , 'Type': len(f_path)*re.findall('\d{1,2}.*',t)}))
    print(re.findall('\d{1,2}.*',t)[0])

#將類型做編碼
file_info['Tcode'] = pd.Categorical(file_info.Type).codes
#將類型編碼存成txt
for ind, typ in enumerate(pd.Categorical(file_info.Type).categories):
    print(ind,typ)
    with open( args.savepath +'/Type_code_reference.txt','a') as f:
        f.writelines([str(ind),' ',typ ,'\n'])
        
file_info.to_csv(args.savepath + '/total_data_ind.csv', index=False)

