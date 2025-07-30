#!/bin/bash
#SBATCH --account=def-skrishna
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=4  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=16000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-00:30
#SBATCH --output=set_GEO18thred-%j.out
#SBATCH --mail-user=yiqing.zhu2@mail.mcgill.ca
#SBATCH --mail-type=BEGIN,END,FAIL

# mem maybe 64000M for total compute 
module purge
module load StdEnv/2020
module load gcc/9.3.0
module load cuda/11.4
module load python/3.10
module load opencv/4.6.0

# remember to use the last zip file
cp ~/projects/rrg-skrishna/yzhu439/CLIP_yq/Tmp.zip $SLURM_TMPDIR
unzip $SLURM_TMPDIR/Tmp.zip -d $SLURM_TMPDIR/

# env_requirements.txt has tested from test4.sh and then get this, in logic this txt would work 
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r $SLURM_TMPDIR/requirements.txt

# change current layer to $SLURM_TMPDIR
cd $SLURM_TMPDIR
unzip $SLURM_TMPDIR/COCOSearch18-images-TP.zip -d $SLURM_TMPDIR/
python pipelines/Set_18thresholdsGEO.py.py
# copy the output csv and image resutls store back to my local place 
cp -r output ~/projects/rrg-skrishna/yzhu439/CLIP_yq/