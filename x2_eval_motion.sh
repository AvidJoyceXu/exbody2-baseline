#!/bin/bash
# Evaluation script for X2 motion tracking performance
# Runs evaluations with different configurations: with/without disturbances, single motion/motion switch

export LD_LIBRARY_PATH=/home/descfly/miniconda3/envs/humanoid/lib/python3.8/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/descfly/miniconda3/envs/humanoid/lib/python3.8/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

cd ~/Lingyun/exbody2/legged_gym/legged_gym

# Configuration
TASK="x2_mimic_priv_distill"
EXPTID="000-01-new"
PROJ_NAME="x2_mimic_priv"
DEVICE="cuda:0"
NUM_EPISODES=10
WINDOW_LENGTH=8

# Paths
BASE_DIR="/home/descfly/Lingyun/exbody2"
DOMAIN_RAND_NO_DR="${BASE_DIR}/humanoidverse/config/domain_rand/x2t2_23dof_nodr.yaml"
TERRAIN_PLANE="${BASE_DIR}/humanoidverse/config/terrain/terrain_locomotion_plane.yaml"
TERRAIN_EVAL="${BASE_DIR}/humanoidverse/config/terrain/terrain_locomotion_eval.yaml"

# Output directory
OUTPUT_DIR="${BASE_DIR}/legged_gym/logs/${PROJ_NAME}/${EXPTID}/eval_results"
mkdir -p ${OUTPUT_DIR}

echo "=========================================="
echo "X2 Motion Tracking Evaluation"
echo "=========================================="
echo "Task: ${TASK}"
echo "Experiment ID: ${EXPTID}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Function to run evaluation
run_eval() {
    local eval_flag=$1
    local dist_type=$2
    local output_name=$3
    
    echo "----------------------------------------"
    echo "Running: ${output_name}"
    echo "----------------------------------------"
    
    python scripts/eval_x2_motion.py \
        --task ${TASK} \
        --exptid ${EXPTID} \
        --proj_name ${PROJ_NAME} \
        --device ${DEVICE} \
        --domain_rand_config ${DOMAIN_RAND_NO_DR} \
        --terrain_config ${TERRAIN_PLANE} \
        --num_episodes ${NUM_EPISODES} \
        --window_length ${WINDOW_LENGTH} \
        --output_path ${OUTPUT_DIR}/${output_name}.json \
        ${eval_flag}
    
    if [ $? -eq 0 ]; then
        echo "✓ Completed: ${output_name}"
    else
        echo "✗ Failed: ${output_name}"
    fi
    echo ""
}

# 1. Single motion evaluation - No disturbances
echo "1. Single Motion Evaluation (No Disturbances)"
run_eval "--eval_single" "nodist" "eval_single_nodist"

# 2. Single motion evaluation - With disturbances  
echo "2. Single Motion Evaluation (With Disturbances)"
python scripts/eval_x2_motion.py \
    --task ${TASK} \
    --exptid ${EXPTID} \
    --proj_name ${PROJ_NAME} \
    --device ${DEVICE} \
    --domain_rand_config ${DOMAIN_RAND_NO_DR} \
    --terrain_config ${TERRAIN_EVAL} \
    --num_episodes ${NUM_EPISODES} \
    --window_length ${WINDOW_LENGTH} \
    --output_path ${OUTPUT_DIR}/eval_single_dist.json \
    --eval_single

# 3. Motion switch evaluation - No disturbances
echo "3. Motion Switch Evaluation (No Disturbances)"
run_eval "--eval_switch" "nodist" "eval_switch_nodist"

# 4. Motion switch evaluation - With disturbances
echo "4. Motion Switch Evaluation (With Disturbances)"
python scripts/eval_x2_motion.py \
    --task ${TASK} \
    --exptid ${EXPTID} \
    --proj_name ${PROJ_NAME} \
    --device ${DEVICE} \
    --domain_rand_config ${DOMAIN_RAND_NO_DR} \
    --terrain_config ${TERRAIN_EVAL} \
    --num_episodes ${NUM_EPISODES} \
    --window_length ${WINDOW_LENGTH} \
    --output_path ${OUTPUT_DIR}/eval_switch_dist.json \
    --eval_switch

echo "=========================================="
echo "All evaluations completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=========================================="

# Generate summary report
echo ""
echo "Generating summary report..."
python - << EOF
import json
import os
from pathlib import Path

output_dir = "${OUTPUT_DIR}"
files = [
    "eval_single_nodist.json",
    "eval_single_dist.json", 
    "eval_switch_nodist.json",
    "eval_switch_dist.json"
]

summary = {}
for fname in files:
    fpath = os.path.join(output_dir, fname)
    if os.path.exists(fpath):
        with open(fpath, 'r') as f:
            data = json.load(f)
            summary[fname.replace('.json', '')] = data

summary_path = os.path.join(output_dir, "summary.json")
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"Summary saved to: {summary_path}")
EOF

echo ""
echo "Done!"

