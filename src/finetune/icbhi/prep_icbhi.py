# -*- coding: utf-8 -*-
'''
Created on Thu Jul 27 18:49:44 2023

@author: SNUH
'''

import os
import ssl
import json
import wget
import zipfile
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import resample

ssl._create_default_https_context = ssl._create_unverified_context

def create_json_file(base_path, dataset):    
    # Create a list to store the data for the JSON file
    json_data = []
    
    for basename in dataset:
        # Adding an extension to a basename
        filename = basename + '.wav'
        
        # Read the corresponding '.txt' file
        data = pd.read_csv(base_path + basename + '.txt', sep='\t', header=None)
        
        # Load the '.wav' file
        sampling_rate, audio_data = wavfile.read(base_path + filename)
        
        # Process each row in the '.txt' file
        for index, row in data.iterrows():
            # Determine the label
            if row[2] == 0 and row[3] == 0:
                label = 'normal'
            elif row[2] == 1 and row[3] == 1:
                label = 'crackle&wheezing'
            elif row[2] == 1:
                label = 'crackle'
            elif row[3] == 1:
                label = 'wheezing'
            
            # Split and resample the audio data
            start_sample = int(row[0] * sampling_rate)
            end_sample = int(row[1] * sampling_rate)
            segment = audio_data[start_sample:end_sample]
            resampled = resample(segment, int(len(segment) * 16000 / sampling_rate))
            
            # Save the resampled segment as a '.wav' file
            segment_filename = basename + '_segment_' + str(index + 1) + '.wav'
            segment_path = base_path + 'segment/' + segment_filename
            wavfile.write(segment_path, 16000, resampled.astype(np.int16))
            
            # Add an entry to the JSON data
            json_data.append({'wav': segment_path, 'labels': label})

    return json_data

# Download ICBHI Database & official_split.txt
if os.path.exists('./data/ICBHI_final_database') == False:
    ichbi_url = 'https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip'
    wget.download(ichbi_url, out='./data/')
    with zipfile.ZipFile('./data/ICBHI_final_database.zip', 'r') as zip_ref:
        zip_ref.extractall('./data/')
    os.remove('./data/ICBHI_final_database.zip')
    
    txt_url = 'https://raw.githubusercontent.com/raymin0223/patch-mix_contrastive_learning/main/data/icbhi_dataset/official_split.txt'
    wget.download(txt_url, out='./data/')


# Create train/eval dataset according to official_split.txt
lines = open('./data/official_split.txt').read().splitlines()
train_set = []
eval_set = []

for line in lines:
    basename, fold = line.strip().split('\t')
    if fold == 'train':
        train_set.append(basename)
    else:
        eval_set.append(basename)

# Generate JSON data by train/eval
base_path = os.path.abspath(os.getcwd()) + '/data/ICBHI_final_database/'
if os.path.exists('./data/ICBHI_final_database/segment/') == False:
    os.mkdir('./data/ICBHI_final_database/segment/')
train_json_data = create_json_file(base_path, train_set)
eval_json_data = create_json_file(base_path, eval_set)

print('{:d} training samples, {:d} test samples'.format(len(train_json_data), len(eval_json_data)))

# Save the JSON data to a file
with open('./data/icbhi_train.json', 'w') as f:
    json.dump({'data': train_json_data}, f, indent=2)

with open('./data/icbhi_eval.json', 'w') as f:
    json.dump({'data': eval_json_data}, f, indent=2)
    
# Create csv file
data = [
    [0, 'normal', 'normal'],
    [1, 'crackle&wheezing', 'crackle&wheezing'],
    [2, 'crackle', 'crackle'],
    [3, 'wheezing', 'wheezing']
]
columns = ['index', 'mid', 'display_name']
class_labels_indice = pd.DataFrame(data, columns=columns)
class_labels_indice.to_csv('./data/icbhi_class_labels_indices.csv', index=False)
    
# Replace <your_directory> with the path to the directory containing your '.wav' and '.txt' files. This code will generate a '.json' file for each '.wav' file in the directory. Each '.json' file will contain a list of entries for the segments of the corresponding '.wav' file, with the filename of the resampled segment and its label.

# The code also generates a new '.wav' file for each segment of the original '.wav' file. The new files are resampled to 16kHz and named <original_basename>_segment_<n>.wav, where <original_basename> is the basename of the original '.wav' file and <n> is the segment number.

# Please note that the resampling process might lead to information loss due to the significant reduction in the sampling rate. Also, you need to have the scipy and pandas libraries installed in your Python environment to run this code.