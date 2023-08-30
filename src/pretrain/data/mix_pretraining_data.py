# -*- coding: utf-8 -*-
# combine ICBHI, HF_Lung_V1 and chest_wall, count how many samples of each of them.

import json
import random

def combine_json(file_list, name='icbhi_hflungv1_chestwall'):
    wav_list = []
    for file in file_list:
        with open(file, 'r') as f:
            cur_json = json.load(f)
        cur_data = cur_json['data']
        print(len(cur_data))
        random.shuffle(cur_data)
        for entry in cur_data:
            entry['labels'] = 'normal'

        wav_list = wav_list + cur_data
    with open(name + '.json', 'w') as f:
        print(len(wav_list))
        json.dump({'data': wav_list}, f, indent=1)


if __name__ == '__main__':
    icbhi_data = '/workspace/pj_resp/ssast/src/pretrain/data/icbhi/icbhi_all.json'
    HF_Lung_V1_data = '/workspace/pj_resp/ssast/src/pretrain/data/HF_Lung_V1/HF_Lung_V1.json'
    chest_wall_data = '/workspace/pj_resp/ssast/src/pretrain/data/chest_wall/chest_wall.json'
    combine_json([icbhi_data, HF_Lung_V1_data, chest_wall_data], name='icbhi_hflungv1_chestwall')