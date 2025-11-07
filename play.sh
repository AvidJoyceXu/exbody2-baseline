export LD_LIBRARY_PATH=/home/descfly/miniconda3/envs/humanoid/lib/python3.8/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/descfly/miniconda3/envs/humanoid/lib/python3.8/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
cd ~/Lingyun/exbody2/legged_gym/legged_gym
python scripts/play_priv.py --task g1_mimic_priv 000-00 --motion_name motions_dance_release.yaml --device cuda:0
