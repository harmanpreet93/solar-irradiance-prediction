#!/bin/bash
#SBATCH --time=60:00
#SBATCH --gres=gpu:k20:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4000M

rm -rf ../data/evaluator_script_val_cnn
date
source /project/cq-training-1/project1/teams/team08/server_env/bin/activate

date
echo ~~~~~~~~~~~evaluating now
python ../code/evaluator.py ../log/output.txt ../val_cfg_local.json -u="../code/eval_user_cfg_cnn.json" -s="../log/output_stats.txt"