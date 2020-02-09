#!/bin/bash
#SBATCH --time=60:00
#SBATCH --gres=gpu:k80:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10000M

pwd
module load python/3.7
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index -r requirements.txt
# pip install --no-index tensorflow_gpu

# python ./tensorflow-test.py
./run_training_loop.sh
./run_evaluator.sh
sleep -10
