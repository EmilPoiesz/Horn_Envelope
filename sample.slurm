#!/bin/bash
#SBATCH --job-name=horn_learner
#SBATCH --account=ec30
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4G
#
# By default, request four CPU cores (NumPy may just know how to take
# advantage of them; for larger computations, maybe use between
# six and ten; at some point, we will look at how to run on GPUs
#
#SBATCH --cpus-per-task=4

# NB: this script should be run with "sbatch sample.slurm"!
# See https://www.uio.no/english/services/it/research/platforms/edu-research/help/fox/jobs/submitting.md

source ~/.bashrc

# sanity: exit on all errors and disallow unset environment variables
set -o errexit
set -o nounset

# the important bit: unload all current modules (just in case) and load only the necessary ones
module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load Python/3.10.8-GCCcore-12.2.0
module load SciPy-bundle/2024.05-gfbf-2024a
module load Transformers/4.39.3-gfbf-2023a
module load sympy/1.12-gfbf-2023a


python3 masked_gender_model_script.py ${@}
