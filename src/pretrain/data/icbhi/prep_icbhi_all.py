# -*- coding: utf-8 -*-
import os
import sys
import time
import json

# combine json files
def combine_json(file_list, name='icbhi_all'):
    wav_list = []
    for file in file_list:
        with open(file + '.json', 'r') as f:
            cur_json = json.load(f)
        wav_list = wav_list + cur_json['data']
    with open(name + '.json', 'w') as f:
        json.dump({'data': wav_list}, f, indent=1)
    
path = '/workspace/pj_resp/ssast/src/pretrain/data/icbhi/'
combine_json(['icbhi_train', 'icbhi_eval',], name='icbhi_all')