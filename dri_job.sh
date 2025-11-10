#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem=96G      # increase as needed
#SBATCH --time=48:00:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=24
#SBATCH --mail-user=pcarnah@uwo.ca
#SBATCH --mail-type=ALL

tar --use-compress-program=pigz -xf ~/projects/def-chene/pcarnah/YCB-V-DS.tar.gz -C $SLURM_TMPDIR &
pid=$!

echo "Tar extract background process with PID $pid started."

module load python/3.12 cuda cudnn gcc opencv/4.12
python -m venv $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --upgrade pip
pip install torch==2.7.1 torchvision torchaudio wheel Cython
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
pip install -r requirements.txt
# pip install pillow --force-reinstall


# Wait for the specific background process to finish
echo "Waiting for process $pid to complete..."
wait $pid

export OMP_NUM_THREADS=12
export BLIS_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=12
export MKL_NUM_THREADS=12

python ./core/gdrn_stereo_modeling/main_gdrn.py --no-wandb --config-file "./configs/gdrn_stereo/denstereo/all_double.py"
