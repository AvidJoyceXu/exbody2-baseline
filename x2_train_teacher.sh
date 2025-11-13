export LD_LIBRARY_PATH=/home/descfly/miniconda3/envs/humanoid/lib/python3.8/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/descfly/miniconda3/envs/humanoid/lib/python3.8/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

cd legged_gym
python legged_gym/scripts/train.py \
    --task x2_mimic_priv 000-00-xml \
    --motion_name foo.yaml \
    --device cuda:0 \
    --entity humanoid-2025 \
