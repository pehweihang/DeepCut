#!/bin/bash

##Job Script

#SBATCH --partition=SCSEGPU_UG
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=10G
#SBATCH --job-name=DC
#SBATCH --output=/home/FYP/pehw0013/CZ4045_Proj/slurm_out/output_%x_%j.out
#SBATCH --error=/home/FYP/pehw0013/CZ4045_Proj/slurm_out/error_%x_%j.err

module load anaconda
source activate DeepCut
python src/main.py
