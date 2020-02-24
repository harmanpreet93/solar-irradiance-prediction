#!/bin/bash
#SBATCH --time=15:00
#SBATCH --gres=gpu:k80:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10000M
##SBATCH --reservation=IFT6759_2020-02-21

cp -r data/* $SLURM_TMPDIR/
module load python/3.7
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index -r requirements.txt

./run_training_loop.sh
# ./run_evaluator.sh
