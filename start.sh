#!/bin/sh

#SBATCH -J  wghtest10
#SBATCH -c 8 -N1 -n1 -p GPU-8A100 -\-gres=gpu:2 --qos=gpu_8a100

# conda activate openmmlab
# module load cuda/11.7.1_515.65.01

# Go to the job submission directory and run your application
cd $HOME

path=$(pwd)
#Execute applications in parallel
# python  /gpfs/home/sklfs/zhongw/mmSum/goggle/mmdetection-master/tools/train.py  "$@"
echo $path
# python /home/sklfs/malloc/documents/code/python/reproduce/Pytorch-UNet/train.py "$@"
# python /home/qixinggroup/wgh123/Project/DenoisingDiffusionProbabilityModel-ddpm-/Main.py "$@"
accelerate launch --mixed_precision="no" /home/qixinggroup/wgh123/Project/ModelTest/train.py \
  --pretrained_model_name_or_path="/home/qixinggroup/wgh123/Model/SD-2-Inpainting/" \
  --train_data_dir="/home/qixinggroup/wgh123/Dataset/SmokeData/Pair_Data_64450/train/" \
  --dataloader_num_workers=8 \
  --resolution=512 \
  --train_batch_size=32 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=3000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir="/home/qixinggroup/wgh123/Project/ModelTest/AblationModel/ResnetAblation/resnet50/" \
  --mask_random_difference_loss=0.4 \
  --checkpointing_steps=1000 \
  --seed=1337
  "$@"
