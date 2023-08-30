# -*- coding: utf-8 -*-
import os
import sys
import time
import json

def prep(paths, targetpath, name, resample=True):
    if os.path.exists(targetpath) == False:
        os.mkdir(targetpath)

    sample_cnt = 0
    wav_list = []
    begin_time = time.time()
    
    for basepath in paths:
        pathdata = os.walk(basepath)
        for root, dirs, files in pathdata:
            for file in files:
                if file.endswith('.wav'):
                    sample_cnt += 1

                    if resample:
                        # convert all samples to 16kHZ
                        os.system('sox ' + basepath + file + ' -r 16000 ' + targetpath + file + ' > /dev/null 2>&1')

                    # give a dummy label of 'audio' ('Normal' in ICBHI label ontology) to all samples
                    # the label is not used in the pretraining, it is just to make the dataloader.py satisfy.
                    cur_dict = {'wav': targetpath + file, 'label': 'normal'}
                    wav_list.append(cur_dict)

                    if sample_cnt % 1000 == 0:
                        end_time = time.time()
                        print('find {:d}k .wav files, time eclipse: {:.1f} seconds.'.format(int(sample_cnt/1000), end_time-begin_time))
                        begin_time = end_time

    print(sample_cnt)
    with open(name + '.json', 'w') as f:
        json.dump({'data': wav_list}, f, indent=1)
        
if os.system("which sox") != 0:
    print("sox is not installed.")
    sys.exit(1) 
    
paths = ['/workspace/pj_resp/ssast/src/pretrain/data/HF_Lung_V1/test/', '/workspace/pj_resp/ssast/src/pretrain/data/HF_Lung_V1/train/']
targetpath = '/workspace/pj_resp/ssast/src/pretrain/data/HF_Lung_V1/audio_16k/'
name = 'HF_Lung_V1'
prep(paths, targetpath, name, resample=False)