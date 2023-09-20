#!/bin/bash
#SBATCH -p sm
#SBATCH -x sls-sm-1,sls-2080-[3],sls-1080-3,sls-sm-5
##SBATCH -p gpu
##SBATCH -x sls-titan-[0-2]
#SBATCH --gres=gpu:4
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=48000
#SBATCH --job-name="ssast_pretrain"
#SBATCH --output=./slurm_log/log_%j.txt

set -x
# comment this line if not running on sls cluster
# . /data/sls/scratch/share-201907/slstoolchainrc
# source /data/sls/scratch/yuangong/sslast2/sslast2/bin/activate
# export TORCH_HOME=../../pretrained_models
mkdir exp
mkdir slurm_log

pretrain_model=SSAST-Base-Patch-400
pretrain_path=/workspace/pj_resp/ssast/src/pretrain/pretrained_models/${pretrain_model}.pth

task=pretrain_joint
mask_patch=400

# icbhi, hflungv1 and chestwall
dataset=ihc
tr_data=/workspace/pj_resp/ssast/src/pretrain/data/icbhi_hflungv1_chestwall.json
te_data=/workspace/pj_resp/ssast/src/pretrain/data/icbhi/icbhi_eval.json
dataset_mean=-4.2677393
dataset_std=4.5689974
target_length=1024
num_mel_bins=128

model_size=base
# no patch split overlap
fshape=16
tshape=16
fstride=${fshape}
tstride=${tshape}
# no class balancing as it implicitly uses label information
bal=none
batch_size=8
lr=1e-4
# learning rate decreases if the pretext task performance does not improve on the validation set
lr_patience=2
epoch=100
# no spectrogram masking
freqm=0
timem=0
# no mixup training
mixup=0

exp_dir=./exp/230920_mask01-${model_size}-f${fshape}-t${tshape}-b$batch_size-lr${lr}-m${mask_patch}-${task}-${dataset}

CUDA_CACHE_DISABLE=1 python -W ignore ../run.py --dataset ${dataset} \
--data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
--label-csv ./data/icbhi_class_labels_indices.csv \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride --fshape ${fshape} --tshape ${tshape} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} --num_mel_bins ${num_mel_bins} \
--model_size ${model_size} --mask_patch ${mask_patch} --n-print-steps 100 \
--pretrained_mdl_path ${pretrain_path} \
--task ${task} --lr_patience ${lr_patience} --epoch_iter 4000