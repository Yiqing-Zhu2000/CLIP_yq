#!/bin/bash
#SBATCH --account=def-skrishna
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=4  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=16000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-00:30
#SBATCH --output=cliptest2-%j.out
#SBATCH --mail-user=yiqing.zhu2@mail.mcgill.ca
#SBATCH --mail-type=BEGIN,END,FAIL

module load python/3.10  # Make sure to choose a version that suits your application
# remember to use the last zip file
cp ~/projects/rrg-skrishna/yzhu439/CLIP_yq/Tmp.zip $SLURM_TMPDIR
unzip $SLURM_TMPDIR/Tmp.zip -d $SLURM_TMPDIR/

# env_requirements.txt has tested from test4.sh and then get this, in logic this txt would work 
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index torch torchvision
pip install --no-index ftfy regex tqdm
pip install --no-index packaging

# change current layer to Tmp
cd $SLURM_TMPDIR
python yiqing_test.py
python yiqing_test2.py
# copy the output_model store back to my local place 
cp -r patches_output ~/projects/rrg-skrishna/yzhu439/CLIP_yq/